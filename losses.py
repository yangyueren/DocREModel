import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, logits, labels):

        # import pdb; pdb.set_trace()
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        pos_attn = logit1 - logit1[:, 0].unsqueeze(1)
        pos_attn = (1-F.softmax(pos_attn, dim=1)) * 1
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels * pos_attn).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        neg_attn = logit2[:, 0].unsqueeze(1) - logit2
        neg_attn = (1-F.softmax(neg_attn, dim=1)) * 1

        loss2 = torch.sum( -torch.log( (torch.exp(logit2)) / torch.sum( torch.exp(logit2) * neg_attn ) * th_label),   dim=1)

        # original
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
