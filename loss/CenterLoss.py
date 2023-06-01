import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.LabelSmoothing import smooth_cross_entropy_loss

class CenterLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(CenterLoss, self).__init__()
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']                
        self.label_smoothing = options['label_smoothing']
        self.num_classes = options['num_classes']
        # init paramaters
        self.centers = nn.Parameter(torch.Tensor(options['feat_dim'], self.num_classes))
        self.w = nn.Parameter(torch.Tensor(options['feat_dim'], self.num_classes))
        self.b = nn.Parameter(torch.Tensor(1, self.num_classes))
        nn.init.xavier_normal_(self.centers)
        nn.init.xavier_normal_(self.w)
        nn.init.constant_(self.b, 0.)

    def forward(self, x, labels=None):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            y: logits with shape (batch_size).
            labels: ground truth labels with shape (batch_size).
            metric: method to calculate distance.
        """
        logits = x.mm(self.w) + self.b

        if labels is None: 
            return logits, 0

        batch_size = x.size(0)
        # Calculate center loss
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=0, keepdim=True).expand(batch_size, self.num_classes)
        distmat.addmm_(x, self.centers, beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().cuda()
        labels_m = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_m.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        pl_loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        # Option for label smoothing
        if not self.label_smoothing:
            ce_loss = F.cross_entropy(logits / self.temp, labels)
        else:
            ce_loss = smooth_cross_entropy_loss(logits / self.temp, labels=labels, smoothing=self.label_smoothing, dim=-1)

        loss = ce_loss + self.weight_pl * pl_loss

        return logits, loss