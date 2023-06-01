import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.LabelSmoothing import smooth_cross_entropy_loss

class Softmax(nn.Module):
    def __init__(self, **options):
        super(Softmax, self).__init__()
        self.temp = options['temp']
        self.label_smoothing = options['label_smoothing']
        self.w = nn.Parameter(torch.Tensor(options['feat_dim'], options['num_classes']))
        self.b = nn.Parameter(torch.Tensor(1, options['num_classes']))
        nn.init.xavier_normal_(self.w)
        nn.init.constant_(self.b, 0.)

    def forward(self, x, labels=None):
        logits = x.mm(self.w) + self.b
        if labels is None:
            return logits, 0

        if not self.label_smoothing:
            loss = F.cross_entropy(logits / self.temp, labels)
        else:
            loss = smooth_cross_entropy_loss(logits / self.temp, labels=labels, smoothing=self.label_smoothing, dim=-1)

        return logits, loss