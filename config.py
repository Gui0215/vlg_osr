# ----------------------
# PROJECT ROOT DIR
# ----------------------
project_root_dir = '/home/gui/Downloads/gyy0525/'

# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
exp_root = '/home/gui/Downloads/gyy0525/output'  # Directory to store experiment output (checkpoints, logs, etc)
save_dir = '/home/gui/Downloads/gyy0525/eval'    # Evaluation save dir

# Evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = '/home/gui/Downloads/cl_osr/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = '/home/gui/Downloads/cl_osr/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
mnist_root = '/home/gui/Downloads/Datasets/mnist/'                                      # MNIST
svhn_root = '/home/gui/Downloads/Datasets/svhn'                                         # SVHN
cifar_10_root = '/home/gui/Downloads/Datasets/cifar-10'                                 # CIFAR10
cifar_100_root = '/home/gui/Downloads/Datasets/cifar-100'                               # CIFAR100
tin_train_root_dir = '/home/gui/Downloads/Datasets/tiny-imagenet-200/train'             # TinyImageNet Train
tin_val_root_dir = '/home/gui/Downloads/Datasets/tiny-imagenet-200/val/images'          # TinyImageNet Val

cub_root = '/home/gui/Downloads/Datasets/CUB_200_2011/'                                 # CUB
aircraft_root = '/home/gui/Downloads/Datasets/fgvc-aircraft-2013b'                      # FGVC-Aircraft
car_root = "/home/gui/Downloads/Datasets/cars_{}/"                                      # Stanford Cars
meta_default_path = "/work/sagar/datasets/stanford_car/devkit/cars_{}.mat"              # Stanford Cars Devkit
pku_air_root = '/home/gui/Downloads/Datasets/pku-air-300/AIR'                           # PKU-AIRCRAFT-300

imagenet_root = '/home/gui/Downloads/Datasets/imagenet'                                 # ImageNet-1K
imagenet21k_root = '/work/sagar/datasets/imagenet21k_resized_new'                       # ImageNet-21K-P

# ----------------------
# FGVC / IMAGENET OSR SPLITS
# ----------------------
osr_split_dir = '/home/gui/Downloads/cl_osr/data/open_set_splits'

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
imagenet_moco_path = '/home/gui/Downloads/model/moco_v2_800ep_pretrain.pth.tar'
places_moco_path = '/home/gui/Downloads/model/moco_v2_places.pth'
places_supervised_path = '/home/gui/Downloads/model/supervised_places.pth'
imagenet_supervised_path = '/home/gui/Downloads/model/supervised_imagenet.pth'