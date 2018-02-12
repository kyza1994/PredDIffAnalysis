from PDA import *

from vis.visualization import visualize_saliency, overlay, visualize_cam
from vis.utils import utils
from keras import activations
from copy import deepcopy
from skimage.color import rgb2grey


def saliency_map(model, img, class_index=1, backprop_modifier=None):
    """
    Generates an attention heatmap over the `img` for maximizing `class_index`
    output in the last layer.
    Args:
        model: The `keras.models.Model` instance.
        class_index: class indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        img: The model input for which activation map needs to be visualized.
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None). Possible choices: None, 'guided', 'relu'
    """

    layer_idx = -1
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    grads = visualize_saliency(model, layer_idx, filter_indices=class_index, seed_input=img,
                               backprop_modifier=backprop_modifier)
    return rgb2grey(grads).reshape(-1)



def cam_saliency_map(model, img, class_index=1, backprop_modifier=None):
    """
    Generates an attention heatmap over the `img` for maximizing `class_index`
    output in the last layer.
    Args:
        model: The `keras.models.Model` instance.
        class_index: class indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)
            For `keras.layers.Dense` layer, `filter_idx` is interpreted as the output index.
            If you are visualizing final `keras.layers.Dense` layer, consider switching 'softmax' activation for
            'linear' using [utils.apply_modifications](vis.utils.utils#apply_modifications) for better results.
        img: The model input for which activation map needs to be visualized.
        backprop_modifier: backprop modifier to use. See [backprop_modifiers](vis.backprop_modifiers.md). If you don't
            specify anything, no backprop modification is applied. (Default value = None). Possible choices: None, 'guided', 'relu'
    """

    layer_idx = -1
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    grads = visualize_cam(model, layer_idx, filter_indices=class_index, seed_input=img,
                               backprop_modifier=backprop_modifier)
    return rgb2grey(grads).reshape(-1)