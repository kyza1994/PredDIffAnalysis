# -*- coding: utf-8 -*-
"""
Some utility functions for visualisation, not documented properly
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def plot_map(x_test_im, predDiff, class_indx, classnames=None, save_path=None):
    '''
    Plot the results of the relevance estimation
    '''
    imsize = x_test_im.shape

    tarClass = None
    if classnames is not None and class_indx is not None:
        tarClass = classnames[class_indx]
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(x_test_im, cmap='gray')
    plt.title('original')

    plt.subplot(1, 3, 2)
    p = predDiff.reshape((imsize[0], imsize[1], -1))[:, :, class_indx]
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')
    plt.title('weight of evidence')


    plt.subplot(1, 3, 3)
    if tarClass is not None:
        plt.title('class: {}'.format(tarClass))
    plt.imshow(x_test_im, cmap='gray', )
    plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest', alpha=0.5)

    fig = plt.gcf()
    fig.set_size_inches(np.array([8,8]), forward=True)
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()
    plt.show()


    if save_path is not None:
        plt.savefig(save_path)


def plot_all_maps(x_test_im, predDiff, classnames=None,  save_path=None):

    imsize = x_test_im.shape
    prd = predDiff.reshape((imsize[0], imsize[1], -1))

    ncols = 3
    nrows = math.ceil((len(classnames) + 1) / ncols)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 15))

    plt.subplot(nrows, ncols, 1)
    plt.title('original')
    plt.imshow(x_test_im, cmap='gray')

    for i in range(1, 6):
        plt.subplot(nrows, ncols, 2 * i)
        if classnames is not None:
            plt.title('class {}'.format(classnames[2 * i - 2]))
        p = prd[:, :, 2 * (i - 1)]
        plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')

        plt.subplot(nrows, ncols, 2 * i + 1)
        if classnames is not None:
            plt.title('class {}'.format(classnames[2 * i - 1]))
        p = prd[:, :, 2 * (i - 1) + 1]
        plt.imshow(p, cmap=cm.seismic, vmin=-np.max(np.abs(p)), vmax=np.max(np.abs(p)), interpolation='nearest')

    if save_path is not None:
        plt.savefig(save_path)