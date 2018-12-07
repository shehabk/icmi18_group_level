from __future__ import print_function
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
import torchvision.models as torchmodels
##################################################


def main():
    print ('Start Blurred Evaluation')

    args = parse_arguments()
    arch       = args.arch
    # allignment = args.alligned
    trained_on = args.trained_on
    test_data  = args.test_data

    choose_last_model = args.choose_last_model
    if choose_last_model==True:
        arch_pad = '_last'
    else:
        arch_pad = ''

    if choose_last_model == True:
        model_name = 'checkpoint.pth.tar'
    else:
        model_name = 'model_best.pth.tar'

    image_list = 'data/image_lists/global_256/%s.txt' % (test_data)
    pretrained_model = 'models/global_blurred/%s/%s/default/models/%s'%(trained_on , arch , model_name)
    root           = 'data/cropped_images/global_blurred_256'
    store_outputs  = 'data/predictions/global_blurred/%s/%s/%s' %(trained_on,
                                                                                    test_data,
                                                                                    arch + arch_pad)

    # image_list = 'emotiw_2018/partition/global/val.txt'
    # pretrained_model = 'emotiw/group_blurred/vgg16_bn/default/models/model_best.pth.tar'
    # root   = 'emotiw_2018/cropped_images/group_based_blurred_faces_256_80'



    # processed_db = config_util.get_processed_db_dir()
    # exp_db = config_util.get_exp_db_dir()
    image_list_fullpath = os.path.join( project_dir , image_list )
    pretrained_model_fullpath = os.path.join(project_dir , pretrained_model)
    store_outputs_fullpath    = os.path.join(project_dir , store_outputs)
    if not os.path.exists(store_outputs_fullpath):
        os.makedirs(store_outputs_fullpath)

    image_paths , labels = read_imagelist(image_list_fullpath)
    model = get_model(pretrained_model_fullpath)
    model.eval()
    outputs = list()
    filtered_labels = list()
    correct = 0
    miss_count = 0
    for id in range( len(image_paths)):
        image_path = image_paths[id]
        blank_output = np.zeros((1,3) , dtype=np.float32)
        blank_label  = np.ones((1,) , dtype=np.int32  )
        if (is_valid(root , image_path)==True):
            image = process_image(root , image_path)
            image = image.cuda()
            output = model( image )
            output_np = output.data.cpu().numpy()
            outputs.append(output_np)
            filtered_labels.append( np.expand_dims(labels[id], axis = 0))

        else:
            # print ('No')
            miss_count = miss_count + 1
            outputs.append(blank_output)
            filtered_labels.append(blank_label)


    outputs = np.concatenate(outputs, axis=0)
    filtered_labels  = np.concatenate(filtered_labels, axis=0)
    output_global_fl = os.path.join(store_outputs_fullpath , 'output_global_blurred.txt' )
    np.savetxt(output_global_fl, outputs)
    output_label_fl = os.path.join(store_outputs_fullpath , 'labels.txt')
    np.savetxt(output_label_fl ,filtered_labels)
    # np.savetxt('output_global_blurred.txt', outputs)
    # np.savetxt('labels.txt' ,filtered_labels)
    print ( 'Global Blurred Accuracy: ' + str(get_accuracy(outputs , filtered_labels)))
    # conf_mat = get_confusion_matrix(outputs, filtered_labels)
    # print (conf_mat)
    # # print(get_confusion_matrix(outputs, filtered_labels))
    # print(methods_util._buildstr(
    #     np.nan_to_num(100. * conf_mat / conf_mat.sum(axis=1, keepdims=True))))
    print ('End Blurred Evaluation')



def get_model( model_path):

    num_classes = 3
    model = torchmodels.vgg16_bn()
    in_features = model.classifier[6].in_features
    n_module = nn.Linear(in_features, num_classes)
    n_classifier = list(model.classifier.children())[:-1]
    n_classifier.append(n_module)
    model.classifier = nn.Sequential(*n_classifier)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model.cuda()


def is_valid( root , image_path ):

    # processed_db = config_util.get_processed_db_dir()
    image_fullpath = os.path.join(project_dir,root, image_path )
    if os.path.isfile(image_fullpath):
        return True
    else:
        return False


def process_image( root , image_path ):
    # processed_db = config_util.get_processed_db_dir()
    image_fullpath = os.path.join(project_dir,root, image_path )


    image = Image.open(image_fullpath);
    image = image.convert('RGB')

    transform = transforms.Compose([
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    if transform:
        image = transform(image)

    image = image.unsqueeze(0)

    return image

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


def parse_arguments():
    parser          = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.add_argument_group('args', 'experiment specific arguments')

    args.add_argument("--arch", metavar='N', default='vgg16_bn',
                         help='architecture to use')
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