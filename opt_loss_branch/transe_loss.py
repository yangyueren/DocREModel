import torch
import torch.nn as nn
import torch.nn.functional as F


class TransELoss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim//2)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(dim//2, 2)
        self.relation_emb = nn.Embedding(97, dim)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)

    # 使用rs
    '''def forward(self, hs, ts, rs, rels):
        # import pdb; pdb.set_trace()
        pred = self.linear(hs + rs - ts)
        label = (rels[:, 1:].sum(1) > 0).long()
        loss = F.binary_cross_entropy_with_logits(pred.squeeze(), label.float())
        return loss'''
    
    # 使用rs embedding
    def forward(self, hs, rs, ts, rels):
        # import pdb; pdb.set_trace()
        indices = torch.arange(0,97).to(hs.device).repeat(hs.shape[0], 1)
        rel_emb = self.relation_emb(indices)
        hidden = F.leaky_relu(self.dropout(self.linear1(hs.unsqueeze(1) + rs.unsqueeze(1) + rel_emb - ts.unsqueeze(1))))
        pred = self.linear2(hidden).squeeze().reshape(-1,2)

        label = rels.reshape(-1)
        weight = torch.ones_like(label)
        label[(label==0) & (torch.rand_like(label.float())>0.05)] = -100
        loss = self.loss_fn(pred, label)
        return loss

    def get_label(self, hs, ts, rs, pred_rel, margin):
        # import pdb; pdb.set_trace()
        indices = torch.arange(0,97).to(hs.device).repeat(hs.shape[0], 1)
        rel_emb = self.relation_emb(indices)
        hidden = F.leaky_relu(self.dropout(self.linear1(hs.unsqueeze(1) + rs.unsqueeze(1) + rel_emb - ts.unsqueeze(1))))
        pred = self.linear2(hidden)# (batch, 97, 2)

        prob = F.softmax(pred, dim=-1)
        mask = (prob[:,:,1] - prob[:,:,0]) > margin
        pred_rel = pred_rel.long() & mask

        return pred_rel

