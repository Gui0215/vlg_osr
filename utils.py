import os
import errno
import torch
import random
import numpy as np
import inspect


from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime

import math

project_root_dir = "/home/gui/Downloads/gyy0525/"


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmRestartPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Reduce learning rate on plateau and reset every T_restart epochs
    """
    def __init__(self, T_restart, *args, ** kwargs):
        super(WarmRestartPlateau, self).__init__(*args, **kwargs)
        self.T_restart = T_restart
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)

        if self.last_epoch > 0 and self.last_epoch % self.T_restart == 0:
            for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                group['lr'] = lr
            self._reset()


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, warmup_epochs, *args, **kwargs):
        super(CosineAnnealingWarmupRestarts, self).__init__(*args, **kwargs)
        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min

        self.warmup_epochs = warmup_epochs

        # Get target LR after warmup is complete
        target_lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + math.cos(math.pi * warmup_epochs / self.T_i)) / 2

        # Linearly interpolate between minimum lr and target_lr
        linear_step = (target_lr - self.eta_min) / self.warmup_epochs
        self.warmup_lrs = [self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)]

    def step(self, epoch=None):
        # Called on super class init
        if epoch is None:
            super(CosineAnnealingWarmupRestarts, self).step(epoch=epoch)

        else:
            if epoch < self.warmup_epochs:
                lr = self.warmup_lrs[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            else:
                super(CosineAnnealingWarmupRestarts, self).step(epoch=epoch)



def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    St9rip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def init_experiment(args, runner_name=None):
    args.cuda = torch.cuda.is_available()

    if args.device == 'None':
        args.device = torch.device("cuda:0" if args.cuda else "cpu")
    else:
        args.device = torch.device(args.device if args.cuda else "cpu")

    print(args.gpus)

    # Get filepath of calling script
    # if runner_name is None:
    #     runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    # root_dir = os.path.join(args.exp_root, *runner_name)

    # if not os.path.exists(root_dir):
    #     os.makedirs(root_dir)

    # Unique identifier for experiment
    now = '({}.{:02d}.{:02d}|{:02d}.{:02d})'.format(datetime.now().year, datetime.now().month, datetime.now().day,\
                                                    datetime.now().hour, datetime.now().minute)

    log_dir = os.path.join(args.exp_root, 'log', now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    os.mkdir(model_root_dir)

    args.model_dir = model_root_dir

    print(f'Experiment saved to: {args.log_dir}')

    args.writer = SummaryWriter(log_dir=args.log_dir)

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    print(runner_name)
    print(args)

    return args


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_default_hyperparameters(args):

    """
    Adjusts args to match parameters used in paper: https://arxiv.org/abs/2110.06207
    """

    hyperparameter_path = os.path.join(project_root_dir, 'utils/paper_hyperparameters.csv')
    df = pd.read_csv(hyperparameter_path)

    df = df.loc[df['Loss'] == args.loss]
    hyperparams = df.loc[df['Dataset'] == args.dataset].values[0][2:]

    # -----------------
    # DATASET / LOSS specific hyperparams
    # -----------------
    args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = hyperparams

    # -----------------
    # Other hyperparameters
    # -----------------
    args.seed = 0
    args.max_epoch = 600
    args.transform = 'rand-augment'
    args.scheduler = 'cosine_warm_restarts_warmup'
    args.num_restarts = 2
    args.weight_decay = 1e-4

    return args

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_networks(networks, result_dir, name='', loss='', criterion=None):
    mkdir_if_missing(os.path.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Training")
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar-10-10', help="")
    parser.add_argument('--loss', type=str, default='Softmax', help='For cifar-10-100')
    args = parser.parse_args()
    for dataset in ('mnist', 'svhn', 'cifar-10-10', 'cifar-10-100', 'tinyimagenet'):
        args.dataset = dataset
        args = get_default_hyperparameters(args)
        print(f'{dataset}')
        print(args)