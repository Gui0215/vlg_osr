import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
        Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
        It also supports the unsupervised contrastive loss in SimCLR.
    """
    def __init__(self, **options):
        super(SupConLoss, self).__init__()
        self.temperature = options['temperature'] #0.07
        self.contrast_mode = options['contrast_mode'] #'all'
        self.base_temperature = options['base_temperature'] #0.07
        self.use_gpu = options['use_gpu']

    def forward(self, x, labels=None):
        """
            Compute loss for model. If both `labels` and `mask` are None,
            it degenerates to SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf
        Args:
            x: hidden vector of shape (bsz, n_views, ...).
            labels: ground truth of shape (bsz).
        Returns:
            A loss scalar.
        """
        if len(x.shape) != 3:
            raise ValueError('x dim must be 3, such as(bsz, n_views, feature_dim)')

        device = (torch.device('cuda') if self.use_gpu else torch.device('cpu'))

        batch_size = x.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else: 
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of x')
            mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = x.shape[1]
        contrast_feature = torch.cat(torch.unbind(x, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = x[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
