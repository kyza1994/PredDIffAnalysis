# -*- coding: utf-8 -*-
"""

Utility methods for handling the classifiers:
    set_torch_mode(net, gpu)
    get_net(netpath)
    forward_pass(net, x, layer_numbers=[-1], gpu=False)

"""


import torch
from torch.autograd import Variable
import torch.nn as nn


def set_torch_mode(net, gpu):
    ''' Set whether torch runs in gpu or not'''
    if gpu:
        net.cuda()
        
     
def get_net(netpath):

    try:
        net = torch.load(netpath)
    except FileNotFoundError:
        print('Can not load model')
        return None
    return net  
     


def forward_pass(net, x, layer_numbers=[-1], gpu=False):
    ''' 
    Defines a forward pass (modified for our needs) 
    Input:      net            the network
                x              the input, a batch of images
                layer_numbers  for which layers we want to return the output,
                               default is output layer (-1)
                start          in which layer to start the forward pass
    '''
        
    # feed forward the batch through the next
    net.eval()
    if gpu:
        x = x.cuda()
    x = Variable(x, volatile=True)

    if len(layer_numbers) == 1 and layer_numbers[0] == -1:
        returnVals = [net(x)]
    else:
        returnVals = [nn.Sequential(*list(net.children())[:i+1])(x) for i in layer_numbers]
    
    return returnVals

