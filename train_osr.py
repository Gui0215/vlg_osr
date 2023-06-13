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
from torch.utils.tensorboard import SummaryWriter

from methods import train

from utils import WarmRestartPlateaum, CosineAnnealingWarmupRestarts, seed_torch, save_networks
from dataset.open_set_datasets import get_class_splits, get_datasets
from models.model_utils import get_model

import yaml
from datetime import datetime

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/tinyimagenet_resnet18_encoder.yaml",)

    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r") as config_file:
        args = yaml.full_load(config_file)

    return args

def get_optimizer(optimizer_params, params_list):
    if optimizer_params['name'] is None:
        raise NotImplementedError
    elif optimizer_params['name'] == 'SGD':
        optimizer = torch.optim.SGD(params_list, lr=optimizer_params['lr'], momentum=0.9, weight_decay=optimizer_params['weight_decay'])
    elif optimizer_params['name'] == 'Adam':
        optimizer = torch.optim.Adam(params_list, lr=optimizer_params['lr'])
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(n_epochs, optimizer, scheduler_params):
    if scheduler_params['name'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=150)
    elif scheduler_params['name'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50)
    elif scheduler_params['name'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_params['params']['T_max'], 
                                                               eta_min=scheduler_params['params']['eta_min'])
    elif scheduler_params['name'] == 'cosine_warm_restarts':
        if scheduler_params['params']['num_restarts'] is not None:
            num_restarts = scheduler_params['params']['num_restarts']
        else:
            print('Warning: Num restarts not specified...using 2')
            num_restarts = 2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=int(n_epochs / (num_restarts + 1)),
                                                                         eta_min=scheduler_params['params']['eta_min'])

    elif scheduler_params['name'] == 'cosine_warm_restarts_warmup':
        if scheduler_params['params']['num_restarts'] is not None:
            num_restarts = scheduler_params['params']['num_restarts']
        else:
            print('Warning: Num restarts not specified...using 2')
            num_restarts = 2
        scheduler = CosineAnnealingWarmupRestarts(warmup_epochs=10, optimizer=optimizer,
                                                                    T_0=int(n_epochs / (num_restarts + 1)),
                                                                    eta_min=scheduler_params['params']['eta_min'])

    elif scheduler_params['name'] == 'warm_restarts_plateau':
        scheduler = WarmRestartPlateaum(T_restart=120, optimizer=optimizer, threshold_mode='abs', 
                                        threshold=0.5, mode='min', patience=100)

    elif scheduler_params['name'] == 'multi_step':
        if scheduler_params['params']['step'] is not None:
            steps = scheduler_params['params']['step']
        else:
            print('}Warning: No step list for Multi-Step Scheduler, using constant step of 30 epochs')
            steps = [30 * i for i in range(1, 5)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps)

    else:
        raise NotImplementedError

    return scheduler

def get_mean_lr(optimizer):
    return torch.mean(torch.Tensor([param_group['lr'] for param_group in optimizer.param_groups])).item()

def main_worker(options, backbone, dataloaders, optimizer_params, scheduler_params, criterion_params, n_epochs, 
                log_dir, writer):
    # Cuda settings
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic==True
    torch.manual_seed(options['train']['seed'])
    torch.cuda.manual_seed_all(options['train']['seed'])

    # Dataloaders
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    # Model
    print("Creating model: {}".format(backbone))
    net, feat_dim = get_model(backbone)
    net = net.cuda()
    options.update({'feat_dim': feat_dim})

    # Loss
    loss = criterion_params['name']
    loss_class = importlib.import_module('loss.'+loss)
    criterion = getattr(loss_class, loss)(**criterion_params)
    criterion = criterion.cuda()

    # Experiment Settings
    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]
    
    optimizer = get_optimizer(optimizer_params, params_list=params_list)
    scheduler = get_scheduler(n_epochs, optimizer, scheduler_params)

    # Train
    start_time = time.time()
    for epoch in range(n_epochs):
        print("==> Epoch {}/{}".format(epoch+1, n_epochs))
        loss = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        writer.add_scalar('Loss', loss, epoch)
        writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        # Step scheduler
        if scheduler_params['name'] == 'multi_step' or scheduler_params['name'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total time (h:m:s): {}".format(elapsed))
    save_networks(net, log_dir, file_name.split('.')[0]+'_{}'.format(n_epochs), loss, criterion=criterion)

    return results

if __name__ == '__main__':
    options = parse_config()

    # Get hyperparams
    dataset = options['dataset']['dataset']
    img_size = options['dataset']['image_size']

    backbone = options['model']['backbone']

    transform = options['aug']['transform']
    m = options['aug']['rand_aug_m']
    n = options['aug']['rand_aug_n'] 

    amp = options['train']['amp']
    ema = options['train']['ema']
    ema_decay_per_epoch = options['train']['ema_decay_per_epoch']

    seed = options['train']['seed']
    n_epochs = options["train"]["n_epochs"]
    target_metric = options['train']['target_metric']
    stage = options['train']['stage']

    batch_size = options['dataloaders']['batch_size']
    num_workers = options['dataloaders']['num_workers']
    optimizer_params = options["optimizer"]
    scheduler_params = options["scheduler"]
    criterion_params = options["criterion"]

    log_dir = options['log']['dir']
    # Seed
    seed_torch(seed)

    for i in range(5):
        split_idx = i 

        # Prepare data
        train_classes, open_set_classes = get_class_splits(dataset, split_idx)
        datasets = get_datasets(dataset, transform=transform, train_classes=train_classes, open_set_classes=open_set_classes, 
                                image_size=img_size, seed=seed)

        # Randaug hyperparameters
        if transform == 'rand-augment':
            if m is not None:
                if n is not None:
                    datasets['train'].transform.transforms[0].m = m
                    datasets['train'].transform.transforms[0].n = n

        # Dataloader
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=batch_size, shuffle=shuffle, sampler=None, num_workers=num_workers)

        # Save parameters
        options.update(
            {
                'item':     i,
                'known':    train_classes,
                'unknown':  open_set_classes,
                'num_classes': len(train_classes)
            }
        )

        log_name = '{}_{}_{}'.format(dataset, backbone, criterion_params['name'])
        log_dir = os.path.join(log_dir, log_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        time_now = '({}.{:02d}.{:02d}|{:02d}.{:02d})'.format(datetime.now().year, datetime.now().month, datetime.now().day, 
                                                             datetime.now().hour, datetime.now().minute)
        
        writer_dir = os.path.join(log_dir, time_now)
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)

        writer = SummaryWriter(log_dir=writer_dir)

        if dataset== 'cifar-10-100':
            if options['dataset']['out_num'] is not None:
                file_name = '{}_{}.csv'.format(dataset, options['dataset']['out_num'])
            else:
                raise NotImplementedError
        else:
            file_name = dataset + '.csv'

        print('result path:', os.path.join(log_dir, file_name))

        # Main
        res = main_worker(options, backbone, dataloaders, optimizer_params, scheduler_params, criterion_params, n_epochs, 
                          log_dir, writer)

        print("Finished training!")
