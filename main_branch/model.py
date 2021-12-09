import torch
import torch.nn as nn
from opt_einsum import contract
from main_branch.long_seq import process_long_input
# from main_branch.losses import ATLoss
from losses import balanced_loss as ATLoss
from main_branch.transe_loss import TransELoss
import copy
import torch.nn.functional as F

class DocREModel(nn.Module):
    def __init__(self, config, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels) # config.num_labels is 97

        self.attn1 = nn.Linear(emb_size*2, 1)
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
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        
        n, h, _, c = attention.size() # b*layernum*seqlen*seqlen, 每层 word之间的attn
        hss, tss, rss = [], [], []
        h_to_entity_pos_index, t_to_entity_pos_index = [], []
        # entity_pos[0][8] 是一个e -> [(72, 78), (72, 78)]，放了他的mentions
        for i in range(len(entity_pos)): # 对每一篇文章处理
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            # e_emb.append(torch.mean(sequence_output[i, start + offset:end+offset], dim=0)) # yyybug
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
                        # e_emb = torch.mean(sequence_output[i, start + offset:end+offset], dim=0) # yyybug
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
            # rs_y = torch.matmul(ht_att,sequence_output[i])
            # import pdb; pdb.set_trace()
            # assert (rs==rs_y).all(), 'contract error'
            hss.append(hs) 
            tss.append(ts)
            rss.append(rs) # rs是 这对pair的表示，用来预测这对pair的关系
        hss_list = hss
        tss_list = tss
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss_list = rss
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, hss_list, rss_list, tss_list

    def update_ht(self, input_ids, sequence_output, entity_pos, hts, hss_list, rss_list, tss_list):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        
        c = sequence_output.shape[1]
        hss, tss = [], []
        # entity_pos[0][8] 是一个e -> [(72, 78), (72, 78)]，放了他的mentions
        for i in range(len(entity_pos)):
            entity_embs = [] # 处理一篇文章
            # pair vs  sentence  -> attn
            # get sentence pos
            device = (input_ids).device
            sentence_pos = torch.cat([torch.tensor([0]).to(device), (input_ids[i] == 102).nonzero(as_tuple=True)[0] ],dim=0) # [CLS] + [SEP, SEP, SEP...]
            sent_emb = sequence_output[i][sentence_pos]
            # pair vs  sentence  -> attn
            
            
            d = sent_emb.shape[0]
            b = sent_emb.repeat(rss_list[i].shape[0], 1)
            a = rss_list[i].unsqueeze(1).repeat(1, d, 1).reshape(-1, b.shape[-1])
            pair_sent_attn = F.leaky_relu(self.attn1(torch.cat([a,b], dim=1)).reshape(-1, d))
            pair_sent_attn = F.softmax(pair_sent_attn+1e-8,dim=0)


            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            
            
            for j in range(len(ht_i)): # 一个pair
                h_mens = entity_pos[i][ht_i[j][0]]
                t_mens = entity_pos[i][ht_i[j][1]]

                def get_mentions_embedding_and_sentid(h_mens):
                    def get_sentid(pos):
                        if pos == 0:
                            return 0
                        for idx, sp in enumerate(sentence_pos):
                            if pos > sp:
                                return idx
                        return 0
                    
                    if len(h_mens) > 1:
                        e_emb, mention2sentids = [], []
                        for start, end in h_mens:
                            if start + offset < c:
                                # In case the entity mention is truncated due to limited max seq length.
                                # e_emb.append(sequence_output[i, start + offset])
                                e_emb.append(torch.mean(sequence_output[i, start + offset:end+offset], dim=0)) # yyybug
                                mention2sentids.append(get_sentid(start+offset))
                        if len(e_emb) > 0:
                            e_emb = torch.stack(e_emb, dim=0)
                            mention2sentids = torch.tensor(mention2sentids, dtype=torch.int64)
                        else:
                            e_emb = torch.zeros([1,self.config.hidden_size]).to(sequence_output)
                            
                    else:
                        mention2sentids = []
                        start, end = h_mens[0]
                        if start + offset < c:
                            # e_emb = sequence_output[i, start + offset]
                            e_emb = [torch.mean(sequence_output[i, start + offset:end+offset], dim=0)] # yyybug
                            mention2sentids.append(get_sentid(start+offset))
                            
                        else:
                            e_emb = [torch.zeros(self.config.hidden_size).to(sequence_output)]
                            mention2sentids = [0]
                        e_emb = torch.stack(e_emb, dim=0)
                        mention2sentids = torch.tensor(mention2sentids, dtype=torch.int64)
                    return e_emb.to(sequence_output.device), mention2sentids.to(sequence_output.device)

                # import pdb; pdb.set_trace()
                h_mention_embedding, h_mention_sentid = get_mentions_embedding_and_sentid(h_mens)
                t_mention_embedding, t_mention_sentid = get_mentions_embedding_and_sentid(t_mens)
                

                weighted_h_mentions = torch.matmul(torch.index_select(pair_sent_attn[j], 0, h_mention_sentid).unsqueeze(0), h_mention_embedding)
                hss_list[i][j] += torch.logsumexp(weighted_h_mentions, dim=0)

                weighted_t_mentions = torch.matmul(torch.index_select(pair_sent_attn[j], 0, t_mention_sentid).unsqueeze(0), t_mention_embedding)
                tss_list[i][j] += torch.logsumexp(weighted_t_mentions, dim=0)
            
            
        hss = torch.cat(hss_list, dim=0)
        tss = torch.cat(tss_list, dim=0)
        
        return hss, tss


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
    
        
        hs, rs, ts, hss_list, rss_list, tss_list = self.get_hrt(sequence_output, attention, entity_pos, hts)
        
        # hs, ts = self.update_ht(input_ids, sequence_output, entity_pos, hts, hss_list, rss_list, tss_list)
        


        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        # loss_t = torch.tensor(0.0).to(sequence_output.device)
        # # import pdb; pdb.set_trace()
        # if labels is not None:
        #     rels = [torch.tensor(label) for label in labels]
        #     rels = torch.cat(rels, dim=0).to(sequence_output.device)
        #     loss_t = self.transe(hs, rs, ts, rels) # # yyybug , remove transe
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size) # outer product
        logits = self.bilinear(bl)

        output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output
