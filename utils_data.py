# -*- coding: utf-8 -*-
"""
Utility methods for handling the ImageNet data:
    get_data(path_data)
"""

import numpy as np
import os
import PIL
from torchvision import transforms


def get_data(path_data, transformation=None):
    """
    Returns a small dataset of  data.
    Input:
            path_data         path to data
            transformation    transformation to apply
    Output:
            X                 the feature values, in a matrix
                              (numDatapoints, [imageDimensions])
            X_im              the features as uint8 values, to display
                              using plt.imshow()
            X_filenames       the filenames, with the dots removed
    """

    # get a list of all the images
    img_list = os.listdir(path_data)

    # throw away files that are not in the allowed format (png or jpg)
    for img_file in img_list:
        if not (img_file.endswith(".png") or img_file.endswith(".jpg")):
            img_list.remove(img_file)

    # fill up data matrix
    X = []
    X_filenames = []
    for i in range(len(img_list)):
        img = PIL.Image.open('{}/{}'.format(path_data, img_list[i]))
        if len(X) == 0:
            X = [img]
        else:
            X.append(img)
        X_filenames.append(''.join(img_list[i].split('.')[:-1]))

    # cast to image values that can be displayed directly with plt.imshow()
    X_im = [np.array(img) for img in X]

    # preprocess
    if transformation == None:
        transformation = transforms.Compose([
            transforms.ToTensor()
        ])

    X_pre = []
    for img in X:
        X_pre.append(transformation(img).unsqueeze_(0))
    X = X_pre

    return X, X_im, X_filenames