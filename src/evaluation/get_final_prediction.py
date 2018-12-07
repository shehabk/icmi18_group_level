from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader , sampler
import argparse
#####################

############# other required imports
import os
import sys
import shutil
import numpy as np
import glob
from PIL import Image  # Replace by accimage when ready
# import cv2
# import skimage.io
# import skimage.transform
from multiprocessing import Pool
###################################



########## Project Specific Common ##############
project_dir = os.getcwd()
sys.path.insert(0, project_dir)
#################################################

# from util import config_util
from src.util import methods_util
# from util import multi_transforms
# import datasets.emotiw as ds
# import torchvision.models as torchmodels
##################################################


def main():
    print ('Start Final Prediction')

    args = parse_arguments()
    arch_face = 'resnet_i24_34_resnet_i48_18'
    # arch_face  = 'resnet_i48_18'
    arch_global = 'vgg16_bn'
    # arch = 'resnet_i48_cl_18'
    allignment = args.allignment
    trained_on = args.trained_on
    test_data  = args.test_data

    choose_last_model = args.choose_last_model


    # if heatmap_all == True:
    #     heatmap_pad = '_all'
    #     heatmap_crop_dir = 'group_based_heatmaps_264_94_all'
    # else:
    #     heatmap_pad = ''
    #     heatmap_crop_dir = 'group_based_heatmaps_264_94'



    if choose_last_model==True:
        arch_pad = '_last'
    else:
        arch_pad = ''

    if choose_last_model == True:
        model_name = 'checkpoint.pth.tar'
    else:
        model_name = 'model_best.pth.tar'

    # processed_db = config_util.get_processed_db_dir()

    image_list    =    'data/image_lists/global_256/%s.txt'%(test_data)

    output_faces_path, output_global_path, output_global_heatmap_path, output_global_blurred_path = \
        get_output_paths(allignment, trained_on, test_data, arch_face, arch_global, arch_pad )

    output_global = np.loadtxt(output_global_path)
    output_global_heatmap = np.loadtxt(output_global_heatmap_path)
    output_global_blurred = np.loadtxt(output_global_blurred_path)
    output_faces_average = np.loadtxt(output_faces_path)

    # output_global = softmax( output_global , axis = 1)
    # output_global_heatmap = softmax( output_global_heatmap , axis = 1)
    # output_global_blurred = softmax( output_global_blurred , axis = 1)
    # output_faces_average = softmax( output_faces_average , axis = 1)

    # processed_db = config_util.get_processed_db_dir()
    image_list_fullpath = os.path.join( project_dir , image_list )
    image_paths , labels = read_imagelist(image_list_fullpath)
    #
    # bwg   = 0.20 #0.19
    # bwgb  = 0.00 # 0.08
    # bwgh  = 0.25
    # bwfa  = 0.60 # 0.45
    # #
    # #
    # weighted_output = bwg * output_global \
    #                   + bwgb*output_global_blurred \
    #                   + bwgh * output_global_heatmap \
    #                   + bwfa*output_faces_average
    #
    # print (get_accuracy( weighted_output , labels ))
#
    best_acc = 0.0
    bwg   = 0.0
    bwgb  = 0.0
    bwgh  = 0.0
    bwfa  = 0.0

    for wgh in np.arange(0.0 , 1.0 , .01):
        for wg in np.arange( 0.0 , 1.0 - wgh , .01 ):
            for wgb in np.arange(0.0, 1.0 - wg - wgh, .01):
                wfa = 1.0 -wg - wgh -wgb
                # wfa = 0.0
                # print (wg , wgh , wfa)
                output = wg*output_global + wgh*output_global_heatmap \
                            + wgb*output_global_blurred \
                            + wfa*output_faces_average
                acc = get_accuracy(output , labels)
                if acc >= best_acc:
                    best_acc = acc
                    bwg  = wg
                    bwgb = wgb
                    bwgh = wgh
                    bwfa = wfa

    # bwg   = 0.12
    # bwgb  = 0.05
    # bwgh  = 0.18
    # bwfa  = 0.65


    weighted_output = bwg * output_global \
                      + bwgb*output_global_blurred \
                      + bwgh * output_global_heatmap \
                      + bwfa*output_faces_average
    conf_mat = get_confusion_matrix(weighted_output, labels)

    print('Choosen Weights:')
    print('global_weight: ' + str(bwg))
    print('global_blurred_weight: ' + str(bwgb))
    print('global_blurred_weight: ' + str(bwgh))
    print('faces_weight: ' + str(bwfa))


    # print(get_confusion_matrix(weighted_output, labels))
    print('Confusion Matrix')
    print(methods_util._buildstr(
        np.nan_to_num(100. * conf_mat / conf_mat.sum(axis=1, keepdims=True))))

    print('Accuracy:')
    print(get_accuracy(weighted_output,labels))


    print('End FInal Prediction')
#
#
def get_output_paths(allignment,trained_on ,test_data ,arch_face ,arch_global , arch_pad ):

    # processed_db = config_util.get_processed_db_dir()

    #########################################
    store_outputs_face =    'data/predictions/face' \
                            '/%s/%s/%s/%s'%(allignment,trained_on,test_data,arch_face+arch_pad)
    output_faces_path = os.path.join(project_dir , store_outputs_face , 'output_faces_ensemble.txt')



    store_outputs_global    = 'data/predictions/global' \
                              '/%s/%s/%s' %(trained_on,test_data, arch_global + arch_pad)
    output_global_path      = os.path.join(project_dir, store_outputs_global, 'output_global.txt')




    store_outputs_global_heatmap    = 'data/predictions/global_heatmap' \
                                      '/%s/%s/%s' %(trained_on,test_data,arch_global + arch_pad)
    output_global_heatmap_path      = os.path.join(project_dir, store_outputs_global_heatmap, 'output_global_heatmap.txt')



    store_outputs_global_blurred    = 'data/predictions/global_blurred' \
                                      '/%s/%s/%s' %(trained_on, test_data, arch_global + arch_pad)
    output_global_blurred_path      = os.path.join(project_dir, store_outputs_global_blurred, 'output_global_blurred.txt')



    return output_faces_path, output_global_path , output_global_heatmap_path , output_global_blurred_path

def get_accuracy(outputs, labels):

    assert outputs.shape[0] == labels.shape[0]
    preds = np.argmax(outputs, axis=1)
    correct = (preds == labels).sum()
    total = labels.shape[0]
    acc = 1. * correct / total
    return acc


def read_imagelist( image_list):

    image_paths = []
    labels    = []

    file_path = image_list
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            items = line.split(' ')
            lbls = [int(s) for s in items[1:]]
            image_paths.append(items[0])
            labels.append(np.array(lbls))

    labels = np.array(labels);

    return image_paths , labels[:,0]


def get_confusion_matrix(outputs, labels,
                         num_classes=3):

    # outputs = np.concatenate(outputs, axis = 0)
    # labels  = np.concatenate(labels , axis = 0)

    assert outputs.shape[0] == labels.shape[0]

    bb = [1,0,2]

    conf_mat = np.zeros((num_classes, num_classes))

    preds = np.argmax(outputs, axis=1)

    assert preds.shape == labels.shape
    for index in range(labels.shape[0]):
        conf_mat[ bb[int(labels[index])]][bb[preds[index]]] += 1



    return conf_mat

# https://nolanbconaway.github.io/blog/2017/softmax-numpy
def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p


def parse_arguments():
    parser          = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.add_argument_group('args', 'experiment specific arguments')


    args.add_argument("--allignment", metavar='N', default='alligned',
                         help='alligned')
    args.add_argument("--trained_on", metavar='N', default='train',
                         help='train/train_val')
    args.add_argument("--test_data", metavar='N', default='val',
                         help='val/test')
    args.add_argument("--choose_last_model", help="choose the last model",
                        action="store_true")


    return parser.parse_args()

if __name__=='__main__':
    main()