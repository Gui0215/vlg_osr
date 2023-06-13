from models.vgg import VGG32
from models.resnet import *


def get_model(model):
    if model == 'vgg32':
        model = VGG32()
        feat_dim = 128

    elif model in ['resnet18', 'resnet34', 'resnet50']:
        model_obj, feat_dim = MODEL_DICT[model]
        model = model_obj()
        
    else:
        raise NotImplementedError

    return model, feat_dim
