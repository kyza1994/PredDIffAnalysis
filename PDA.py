from prediction_difference_analysis import PredDiffAnalyser

# utilities
import utils_classifiers as utlC
import utils_sampling as utlS
import utils_visualise as utlV


class PDA:
    def __init__(self, netname, net, samplerData, classnames=None, win_size=5,
                 padding_size=3, gpu=False, batch_size=32, overlapping=True, sample_style='conditional',
                 num_samples=10, layer_numbers=[-1], path_to_params=None, lib='keras'):

        self.netname = netname
        self.net = net
        self.samplerData = samplerData
        self.classnames = classnames
        self.layer_numbers = layer_numbers
        self.win_size = win_size
        self.padding_size = padding_size
        self.gpu = gpu
        self.batch_size = batch_size
        self.overlapping = overlapping
        self.num_samples = num_samples
        self.sampler = None
        self.image_dims = samplerData[0].shape[-2:]
        self.pda = None
        self.lib = lib
        self.path_to_params = path_to_params

        if samplerData[0].ndim == 2:
            self.n_channels = 1
        elif samplerData[0].ndim == 3 and samplerData[0].shape[0] == 3:
            self.n_channels = 3
        elif samplerData[0].ndim == 3 and samplerData[0].shape[0] == 2:
            self.n_channels = 2
        else:
            raise ValueError('Bad input shape')

        utlC.set_mode(self.net, gpu=self.gpu, lib=self.lib)
        # target function (mapping input features to output probabilities)
        self.target_func = lambda x: utlC.forward_pass(self.net, x, self.layer_numbers, self.gpu, self.lib,
                                                       self.n_channels)

        if sample_style == 'conditional':
            if self.n_channels == 1:
                self.sampler = utlS.cond_sampler(X=self.samplerData, win_size=self.win_size,
                                                padding_size=self.padding_size, image_dims=self.image_dims,
                                                netname=self.netname, path_to_params=self.path_to_params)
            else:
                self.sampler = utlS.cond_sampler_nch(X=self.samplerData, win_size=self.win_size,
                                                 padding_size=self.padding_size, image_dims=self.image_dims,
                                                 netname=self.netname, n_channels=self.n_channels,
                                                     path_to_params=self.path_to_params)
        elif sample_style == 'marginal':
            self.sampler = utlS.marg_sampler(self.samplerData)


    def run(self, x):
        """
        Input:
        x    torchTensor
        Output:    weights of evidence
        """

        pda = PredDiffAnalyser(x, self.target_func, self.sampler, num_samples=self.num_samples,
                               batch_size=self.batch_size, n_channels=self.n_channels)
        pred_diffs = pda.get_rel_vect(win_size=self.win_size, overlap=self.overlapping)
        return pred_diffs


    def plot_maps(self, x, pred_diff):

        utlV.plot_all_maps(x, pred_diff, self.classnames)


    def plot(self, x, pred_diff, class_indx):

        utlV.plot_map(x, pred_diff, class_indx)