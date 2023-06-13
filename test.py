import torch
import timm
import os
        

backbone = 'Resnet18'
loss = 'Softmax'
dataset = 'Cifar-10'

log_dir = './log/'
log_name = '{}_{}_{}'.format(dataset, backbone, loss)
log_path = os.path.join(log_dir, log_name)

print(log_path)