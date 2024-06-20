import logging
import os
from urllib import request

import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL


def create_model(num_classes,model_path='./models/ml_decoder/tresnet_l_stanford_card_96.41.pth',model_name='tresnet_l',use_ml_decoder=True,load_head=False):
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

    ####################################################################################
    if use_ml_decoder:
        model = add_ml_decoder_head(model,num_classes=num_classes,num_of_groups=-1,
                                    decoder_embedding=768, zsl=0)
    ####################################################################################
    # loading pretrain model
    model_path = model_path
    if model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        if not load_head:
            if 'model' in state:
                key = 'model'
            else:
                key = 'state_dict'
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict() and 'head.fc' not in k and 'head.decoder' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(state[key], strict=True)

    return model