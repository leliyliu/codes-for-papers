import json
from collections import OrderedDict

from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar
from .dla_simple import SimpleDLA
from .dla import DLA
from .dpn import DPN92, DPN26
from .efficientnet import EfficientNetB0
from .resnet import ResNet18, ResNet50, ResNet34, ResNet101, ResNet152
from .vgg import *
from .shufflenet import *
from .shufflenetv2 import *
from .senet import *
from .resnext import *
from .regnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .cresnet import *


def load_pre_state_dict(model, original_state_dict, key_map=None):
    if not isinstance(key_map, OrderedDict):
        with open('models/weight_keys_map/{}'.format(key_map)) as rf:
            key_map = json.load(rf)
    for k, v in key_map.items():
        if 'num_batches_tracked' in k:
            continue
        else:
            print('{} <== {}'.format(k, v))
            model.state_dict()[k].copy_(original_state_dict[v])
