import torch
import torch.nn as nn
import torch.nn.functional as F


class TransELoss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, hs, ts, rs, rels):
        # import pdb; pdb.set_trace()
        pred = self.linear(hs + rs - ts)
        label = (rels[:, 1:].sum(1) > 0).long()
        loss = F.binary_cross_entropy_with_logits(pred.squeeze(), label.float())
        return loss

