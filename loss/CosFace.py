import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.LabelSmoothing import smooth_cross_entropy_loss

class CosFace(nn.Module):
    """
        reference1: <CosFace: Large Margin Cosine Loss for Deep Face Recognition>
        reference2: <Additive Margin Softmax for Face Verification>
    """
    def __init__(self, **options):
        super(CosFace, self).__init__()
        self.label_smoothing = options['label_smoothing']        
        self.s = options['s']
        self.m = options['m']
        self.w = nn.Parameter(torch.Tensor(options['feat_dim'], options['num_classes']))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, labels=None):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        cos_theta = F.normalize(x, dim=1).mm(self.w)

        if labels is None:
            return cos_theta, 0

        with torch.no_grad():
            d_theta = torch.zeros_like(cos_theta)
            d_theta.scatter_(1, labels.view(-1, 1), -self.m, reduce='add')

        logits = self.s * (cos_theta + d_theta)

        if not self.label_smoothing:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = smooth_cross_entropy_loss(logits, labels=labels, smoothing=self.label_smoothing, dim=-1)

        return logits, loss