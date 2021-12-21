import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from opt_einsum import contract
from dp_prompt_branch.long_seq import process_long_input
from dp_prompt_branch.losses import ATLoss
# from dp_prompt_branch.prompt_loss import PromptLoss
import copy
import torch.nn.functional as F

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        # self.loss_fn_prompt = PromptLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels) # config.num_labels is 97

        # self.transe = TransELoss(emb_size)
        

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels # this num_labels is 4


    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
        sequence_output = output.hidden_states[-1]
        attention = output.attentions[-1]
        # import pdb;pdb.set_trace()
        mask_index = torch.where(input_ids== config.mask_token_id)
        predict_token_id = config.verbalizer.get_all_verbalizer_tokenids()
        prob = output.logits[mask_index][:,predict_token_id]

        return sequence_output, attention, prob.reshape(-1, len(predict_token_id))

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    @autocast()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                ):
        # import pdb;pdb.set_trace()
        
        sequence_output, attention, prob = self.encode(input_ids, attention_mask)

        if self.config.predict_type == 'atlop':
            hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
            hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
            ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
            b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            logits = self.bilinear(bl)

            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)

            if labels is not None:
                labels = [torch.tensor(label) for label in labels]
                labels = torch.cat(labels, dim=0).to(sequence_output.device)
                loss = self.loss_fnt(logits.float(), labels.float())
                output = (loss.to(sequence_output),) + output
            return output
        elif self.config.predict_type == 'prompt':
            output = (self.loss_fnt.get_label(prob, num_labels=self.num_labels),)
            if labels is not None:
                labels = [torch.tensor(label) for label in labels]
                labels = torch.cat(labels, dim=0).to(sequence_output.device)
                loss = self.loss_fnt(prob.float(), labels.float())
                output = (loss.to(sequence_output),) + output
            return output
        else:
            raise NotImplementedError
