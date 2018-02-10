# -*- coding: utf-8 -*-
"""

Utility methods for handling the classifiers:
    set_torch_mode(net, gpu)
    get_net(netpath)
    forward_pass(net, x, layer_numbers=[-1], gpu=False)

"""

def set_mode(net, gpu, lib='keras'):
    ''' Set whether torch runs in gpu or not'''
    if gpu and lib == 'torch':
        net.cuda()
    else:
        pass
        
     
def get_net(netpath, lib='keras'):

    try:
        if lib == 'torch':
            import torch
            net = torch.load(netpath)

        elif lib == 'keras':
            from keras.models import load_model
            net = load_model(netpath)

        else:
            print('Not supported lib')

    except FileNotFoundError:
        print('Can not load model')
        return None
    return net  
     


def forward_pass(net, x, layer_numbers=[-1], gpu=False, lib='keras', n_channels=1):
    ''' 
    Defines a forward pass (modified for our needs) 
    Input:      net            the network
                x              the input, a batch of images
                layer_numbers  for which layers we want to return the output,
                               default is output layer (-1)
                start          in which layer to start the forward pass
    '''


    # feed forward the batch through the next
    if lib == 'torch':

        from torch.autograd import Variable
        import torch.nn as nn
        from torch import from_numpy

        if len(x.shape) != 4:
            x = from_numpy(x.reshape(1, n_channels, x.shape[x.ndim - 2], x.shape[x.ndim - 1])).float()
        else:
            x = from_numpy(x).float()

        net.eval()
        if gpu:
            x = x.cuda()
        x = Variable(x, volatile=True)

        if len(layer_numbers) == 1 and layer_numbers[0] == -1:
            returnVals = [net(x).data.numpy()]
            #print(returnVals[0].shape)
        else:
            returnVals = [nn.Sequential(*list(net.children())[:i+1])(x).data.numpy() for i in layer_numbers]

    elif lib == 'keras':

        from keras.models import Sequential
        from keras import backend as K

        if len(x.shape) != 4:
            if K.image_data_format() == 'channels_first':
                x = x.reshape(1, n_channels, x.shape[x.ndim - 2], x.shape[x.ndim - 1])
            else:
                x = x.reshape(1, x.shape[x.ndim - 2], x.shape[x.ndim - 1], n_channels)
        else:
            if K.image_data_format() == 'channels_first':
                pass
            else:
                x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1])


        if len(layer_numbers) == 1 and layer_numbers[0] == -1:
            returnVals = [net.predict(x, batch_size=x.shape[0])]
        else:
            returnVals = [Sequential(net.layers[:i+1]).predict(x, batch_size=x.shape[0]) for i in layer_numbers]

    else:
        print('Not supported lib')

    return returnVals

