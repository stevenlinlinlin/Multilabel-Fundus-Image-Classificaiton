import logging
import os
from urllib import request

import torch

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_tresnet_model(num_classes,model_path='./models/tresnet/MS_COCO_TRresNet_L_448_86.6.pth',model_name='tresnet_l'):
    """Create a model
    """
    # model_params = {'args': args, 'num_classes': args.num_classes}
    # args = model_params['args']
    model_name = model_name.lower()

    if model_name == 'tresnet_m':
        model = TResnetM(num_classes)
    elif model_name == 'tresnet_l':
        model = TResnetL(num_classes)
    elif model_name == 'tresnet_xl':
        model = TResnetXL(num_classes)
    else:
        print("model: {} not found !!".format(model_name))
        exit(-1)

    state = torch.load(model_path, map_location='cpu')
    filtered_dict = {k: v for k, v in state['model'].items() if
                    (k in model.state_dict() and 'head.fc' not in k)}
    model.load_state_dict(filtered_dict, strict=False)

    return model