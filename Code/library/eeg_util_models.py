import torch
from .CNN_models_vrs1 import *


"""
TFG: Cal afegir tots els models 
"""
def instance_model(MODEL_CONFIG,OPTIMIZER_CONFIG):
    
    # Input Parameters
    lr=OPTIMIZER_CONFIG['lr']
    step_size=OPTIMIZER_CONFIG['step_size']
    MODEL_NAME=MODEL_CONFIG['MODEL_NAME']
    
    projmodule_params=MODEL_CONFIG['projmodule_params']
    convnet_params=MODEL_CONFIG['convnet_params']
    outputmodule_params=MODEL_CONFIG['outputmodule_params']

    if MODEL_NAME == 'CNN_ConcatInput':
        # Concatenated Channels 
        model = CNN_ConcatInput(projmodule_params,convnet_params,outputmodule_params).cuda()
       
    elif MODEL_NAME =='CNN_ProjOut_Conv':
        model = CNN_ProjOut_Conv(projmodule_params,convnet_params,outputmodule_params).cuda()
    
    else:
        print('The selected model does not match!')
        return

    """
    TFG: Es podria considerar fer diferents optimitzadors
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if step_size <= 0:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    return model, optimizer, scheduler


