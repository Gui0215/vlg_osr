import torch
import torch.nn.functional as F
from utils import AverageMeter
from loss.LabelSmoothing import smooth_one_hot

from tqdm import tqdm

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()

    loss_all = 0
    for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):

        data = torch.cat([data[0], data[1]], dim=0)
        batch_size = labels.shape[0]
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x = net(data)
            f1, f2 = torch.split(x, [batch_size, batch_size], dim=0)
            x = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(x, labels)
            loss.backward()
            optimizer.step()
    
        losses.update(loss.item(), data.size(0))
        loss_all += losses.avg

    print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

    return loss_all
