# -*- coding: utf-8 -*-
"""
Some utility functions for visualisation, not documented properly
"""

from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab


def plot_results(x_test, x_test_im, predDiff, tarFunc, classnames, save_path):
    '''
    Plot the results of the relevance estimation
    '''
    imsize = x_test_im.shape


    tarIdx = np.argmax(tarFunc(x_test)[0][0].data.numpy())
    tarClass = classnames[tarIdx]
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(x_test_im, cmap='gray')
    plt.title('original')
    #frame = pylab.gca()
    #frame.axes.get_xaxis().set_ticks([])
    #frame.axes.get_yaxis().set_ticks([])

    plt.subplot(1, 3, 2)
    p = predDiff.reshape((imsize[0], imsize[1],-1))[:, :, tarIdx]
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
    #plt.colorbar()
    #plt.imshow(np.abs(p), cmap=cm.Greys_r)
    plt.title('weight of evidence')
    #frame = pylab.gca()
    #frame.axes.get_xaxis().set_ticks([])
    #frame.axes.get_yaxis().set_ticks([])

    plt.subplot(1, 3, 3)
    plt.title('class: {}'.format(tarClass))
    plt.imshow(x_test_im, cmap='gray', )
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest', alpha=0.5)
    #plt.title('class entropy')
    #frame = pylab.gca()
    #frame.axes.get_xaxis().set_ticks([])
    #frame.axes.get_yaxis().set_ticks([])
    
    fig = plt.gcf()
    fig.set_size_inches(np.array([8,8]), forward=True)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.show()
    #plt.savefig(save_path)
    #plt.close()
