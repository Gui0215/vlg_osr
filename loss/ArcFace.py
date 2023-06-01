import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.LabelSmoothing import smooth_cross_entropy_loss

class ArcFace(nn.Module):
    """ 
        reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, **options):
        super(ArcFace, self).__init__()
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
            theta_m = torch.acos(cos_theta.clamp(-1+1e-5, 1-1e-5))
            theta_m.scatter_(1, labels.view(-1, 1), self.m, reduce='add')
            theta_m.clamp_(1e-5, 3.14159)
            d_theta = torch.cos(theta_m) - cos_theta

        logits = self.s * (cos_theta + d_theta)

        if not self.label_smoothing:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = smooth_cross_entropy_loss(logits, labels=labels, smoothing=self.label_smoothing, dim=-1)

        return logits, loss