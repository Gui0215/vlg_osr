import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from methods import train, test, get_feature

from utils import init_experiment, seed_torch, str2bool, get_default_hyperparameters
from utils import get_scheduler, save_networks
from dataset.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model

import json

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='tinyimagenet', help="")
parser.add_argument('--out_num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=64)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--loss', type=str, default='Softmax')
parser.add_argument('--weight_pl', type=float, default=0.001, help="weight for center loss")
parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing."
                                                                        "No smoothing if None or 0")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--feat_dim', type=int, default=512, help="Feature vector dim, only for classifier32 at the moment")
parser.add_argument('--s', type=float, default=30 , help="param for A-Softmax Loss")
parser.add_argument('--m', type=float, default=4, help="param for A-Softmax Loss")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=None)
parser.add_argument('--rand_aug_n', type=int, default=None)

# misc
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--exp_root', type=str, default='/home/gui/Downloads/gyy0525/output')
parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--use_default_parameters', default=False, type=str2bool,
                    help='Set to True to use optimized hyper-parameters from paper', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_cpu', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')


def get_optimizer(args, params_list):
    if args.optim is None:
        if options['dataset'] == 'tinyimagenet':
            optimizer = torch.optim.Adam(params_list, lr=args.lr)
        else:
            optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=args.lr)
    else:
        raise NotImplementedError
    return optimizer


def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()


def main_worker(options, args):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic ==True
        torch.set_float32_matmul_precision('high')
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataloaders
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    # Model
    print("Creating model: {}".format(options['model']))
    net = get_model(args)

    if use_gpu:
        net = net.cuda()

    feat_dim = args.feat_dim

    # Loss
    options.update({'feat_dim': feat_dim, 'use_gpu':  use_gpu})
    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)
    if use_gpu:
        criterion = criterion.cuda()

    # Experiment Settings
    model_path = os.path.join(args.log_dir, options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]
    
    optimizer = get_optimizer(args=args, params_list=params_list)
    scheduler = get_scheduler(optimizer, args)

    # Train
    start_time = time.time()
    best_auroc, best_oscr = 0, 0
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        loss = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        args.writer.add_scalar('Loss', loss, epoch)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Epoch {}: ACC (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch,
                                                                                              results['ACC'],
                                                                                              results['AUROC'],
                                                                                              results['OSCR']))
            if results['AUROC'] > best_auroc:
                best_auroc = results['AUROC']

            if results['OSCR'] > best_oscr:
                best_oscr = results['OSCR']

            if epoch % options['checkpt_freq'] == 0 or epoch == options['max_epoch'] - 1:
                save_networks(net, model_path, file_name.split('.')[0]+'_{}'.format(epoch), options['loss'], criterion=criterion)

            if epoch == options['max_epoch'] - 1:
                get_feature(net, testloader, outloader, **options)

            args.writer.add_scalar('ACC', results['AUROC'], epoch)
            args.writer.add_scalar('AUROC', results['AUROC'], epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        # Step scheduler
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results['ACC'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total time (h:m:s): {}, best AUROC is {:.3f}, best OSCR is {:.3f}, ".format(elapsed,
                                                                                                 best_auroc,
                                                                                                 best_oscr))

    return results

if __name__ == '__main__':
    args = parser.parse_args()

    # Update parameters with default hyperparameters if specified
    if args.use_default_parameters:
        print('NOTE: Using default hyper-parameters...')
        args = get_default_hyperparameters(args)

    args.epochs = args.max_epoch
    img_size = args.image_size
    results = dict()

    for i in range(1):
        # Init
        if args.feat_dim is None:
            if args.model == 'vgg32':
                args.feat_dim = 128 
            elif args.model == 'resnet18' or args.model == 'resnet34':
                args.feat_dim = 512
            else:
                args.feat_dim = 2048

        # Prepare data
        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx,
                                                                     cifar_plus_n=args.out_num)
        img_size = args.image_size

        args.save_name = '{}_{}_{}'.format(args.model, args.seed, args.dataset)
        runner_name = os.path.dirname(__file__).split("/")[-2:]
        args = init_experiment(args, runner_name=runner_name)

        # Seed
        seed_torch(args.seed)

        # Datasets
        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                                args=args)

        # Randaug hyperparameters
        if args.transform == 'rand-augment':
            if args.rand_aug_m is not None:
                if args.rand_aug_n is not None:
                    datasets['train'].transform.transforms[0].m = args.rand_aug_m
                    datasets['train'].transform.transforms[0].n = args.rand_aug_n

        # Dataloader
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size, shuffle=shuffle, sampler=None, 
                                        num_workers=args.num_workers)

        # Save parameters
        options = vars(args)
        options.update(
            {
                'item':     i,
                'known':    args.train_classes,
                'unknown':  args.open_set_classes,
                'img_size': img_size,
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join('/'.join(args.log_dir.split("/")[:-2]), 'results', dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if options['dataset'] == 'cifar-10-100':
            file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
        else:
            file_name = options['dataset'] + '.csv'

        print('result path:', os.path.join(dir_path, file_name))

        # Main
        res = main_worker(options, args)

        # log
        res['split_idx'] = args.split_idx
        res['unknown'] = args.open_set_classes
        res['known'] = args.train_classes
        res['ID'] = args.log_dir.split("/")[-1]
        results[str(args.split_idx)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name), mode='a', header=False)
