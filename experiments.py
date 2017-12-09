import matplotlib
matplotlib.use('Agg')   

# standard imports
import numpy as np
import time
import os

from torchvision import transforms

# most important script - relevance estimator
from prediction_difference_analysis import PredDiffAnalyser

# utilities
import utils_classifiers as utlC
import utils_data as utlD
import utils_sampling as utlS
import utils_visualise as utlV
import sensitivity_analysis_caffe as SA


# ------------------------ CONFIGURATION ------------------------
# -------------------> CHANGE SETTINGS HERE <--------------------

# pick neural network to run experiment for
netpath = 'mnist_model.pt'
netname = 'mnist'
path_data = './test_data'
classnames = list(map(str, range(0, 10)))

# pick for which layers the explanations should be computed
# (names depend on network, output layer is usually called '-1')
layer_numbers = [-1]

# is torch running in gpu mode?
gpu = False

# pick image indices which are analysed (in alphabetical order as in the ./data folder) [0,1,2,...]
# (if None, all images in './data' will be analysed)
test_indices = 0

# window size (i.e., the size of the pixel patch that is marginalised out in each step)
win_size = 3               # k in alg 1 (see paper)

# indicate whether windows should be overlapping or not
overlapping = True

# settings for sampling 
sample_style = 'conditional' # choose: conditional / marginal
num_samples = 10
padding_size = 2            # important for conditional sampling,
                            # l = win_size+2*padding_size in alg 1
                            # (see paper)

# set the batch size - the larger, the faster computation will be
batch_size = 64


# ------------------------ SET-UP ------------------------

net = utlC.get_net(netpath)

utlC.set_torch_mode(net, gpu=gpu)

# get the data
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

X_test, X_test_im, X_filenames = utlD.get_data(path_data, transformation)

image_dims = X_test_im[0].shape

if not test_indices:
    test_indices = [i for i in range(len(X_test))]

# make folder for saving the results if it doesn't exist
path_results = './results/'
if not os.path.exists(path_results):
    os.makedirs(path_results)          
          
# ------------------------ EXPERIMENTS ------------------------

# target function (mapping input features to output probabilities)
target_func = lambda x: utlC.forward_pass(net, x, layer_numbers, gpu)

if sample_style == 'conditional':
    sampler = utlS.cond_sampler(X=X_test, win_size=win_size, padding_size=padding_size,
                                image_dims=image_dims, netname=netname)
elif sample_style == 'marginal':
    sampler = utlS.marg_sampler(X_test, net)

# for the given test indices, do the prediction difference analysis
for test_idx in test_indices:
      
    # get the specific image (preprocessed, can be used as input to the target function)
    x_test = X_test[test_idx]
    # get the image for plotting (not preprocessed)
    x_test_im = X_test_im[test_idx]
    # prediction of the network
    y_pred = np.argmax(utlC.forward_pass(net, x_test, [-1], gpu)[0][0].data.numpy())
    y_pred_label = y_pred
                           
    # get the path for saving the results
    if sample_style == 'conditional':
        save_path = path_results+'{}_{}_winSize{}_condSampl_numSampl{}_paddSize{}_{}'.format(X_filenames[test_idx],
                                                                                             y_pred_label, win_size,
                                                                                             num_samples, padding_size,
                                                                                             netname)
    elif sample_style == 'marginal':
        save_path = path_results+'{}_{}_winSize{}_margSampl_numSampl{}_{}'.format(X_filenames[test_idx], y_pred_label,
                                                                                  win_size, num_samples, netname)

    if os.path.exists(save_path+'.npz'):
        print('Results for ', X_filenames[test_idx], ' exist, will move to the next image.')
        continue
                 
    print("doing test...", "file :", X_filenames[test_idx], ", net:", netpath, ", win_size:", win_size,
          ", sampling: ", sample_style)

    # compute the sensitivity map
    #layer_name = net.blobs.keys()[-2] # look at penultimate layer (like in Simonyan et al. (2013))
    #sensMap = SA.get_sens_map(net, x_test[np.newaxis], layer_name, np.argmax(target_func(x_test)[-1][0]))

    start_time = time.time()
        
    pda = PredDiffAnalyser(x_test, target_func, sampler, num_samples=num_samples, batch_size=batch_size)
    pred_diff = pda.get_rel_vect(win_size=win_size, overlap=overlapping)
    
    # plot and save the results

    #utlV.plot_results(x_test, x_test_im, pred_diff[0], target_func, classnames, test_idx, save_path)
    #np.savez(save_path, *pred_diff)
    print("--- Total computation took {:.4f} minutes ---".format((time.time() - start_time)/60))