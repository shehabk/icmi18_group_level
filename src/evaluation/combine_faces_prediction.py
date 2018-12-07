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
import re
###################################



########## Project Specific Common ##############
project_dir = os.getcwd()
sys.path.insert(0, project_dir)
#################################################

# from util import config_util
# from util import methods_util
# from util import multi_transforms
# import datasets.emotiw as ds
from src.util import methods_util
import torchvision.models as torchmodels
##################################################


def main():
    print ('Start combining face prediction')

    args = parse_arguments()
    small_arch  = 'resnet_i24_34'
    big_arch    = 'resnet_i48_18'
    store_arch  =  small_arch + '_' +  big_arch

    # arch  = 'resnet_i48_18'
    # arch = 'resnet_i48_cl_18'
    allignment = args.allignment
    trained_on = args.trained_on
    test_data  = args.test_data
    choose_last_model = args.choose_last_model
    if choose_last_model==True:
        arch_pad = '_last'
    else:
        arch_pad = ''




    # face_crop_dir = 'data/cropped_images/aligned_faces'

    start = args.start
    end   = args.end

    image_list    =    'data/image_lists/global_256/%s.txt'%(test_data)
    faces_root    =    'data/cropped_images/aligned_faces'



    small_store_outputs =    'data/predictions/face/%s/%s/%s/%s'%(allignment,
                                                                   trained_on,
                                                                   test_data,
                                                                   small_arch+arch_pad)

    big_store_outputs =    'data/predictions/face//%s/%s/%s/%s'%(allignment,
                                                                   trained_on,
                                                                   test_data,
                                                                   big_arch+arch_pad)

    store_outputs = 'data/predictions/face//%s/%s/%s/%s'%(allignment,
                                                           trained_on,
                                                           test_data,
                                                           store_arch+arch_pad)

    # small_trained_model_root = 'models/faces/%s/%s'%(trained_on , small_arch)
    # big_trained_model_root   = 'models/faces/%s/%s'%(trained_on , big_arch)

    landmarks_root     = 'data/landmarks'




    outputs_ensemble = list()
    filtered_labels_ensemble = list()

    miss_count = 0


    for i in range(start,end+1):
        epad  = 'run'+ str(i)
        print (epad)
        big_store_output_r   = os.path.join( project_dir , big_store_outputs , epad)
        small_store_output_r = os.path.join(project_dir  , small_store_outputs, epad)
        store_output_r       = os.path.join(project_dir  , store_outputs, epad)
        image_list_r   =  os.path.join( project_dir , image_list)
        landmarks_root_r = os.path.join(project_dir , landmarks_root)


        if not os.path.exists(store_output_r):
            os.makedirs(store_output_r)

        image_paths , labels = read_imagelist(image_list_r)
        # model = get_model(trained_model_r , arch = arch)
        # model.eval()

        outputs = list()
        filtered_labels = list()

        miss_count = 0
        for id in range(len(image_paths)):
            image_path = image_paths[id]

            tokens = image_path.split('/')
            image_dir = '/'.join(tokens[:-1])
            image_name = tokens[-1][:-4]
            big_output_dir   = os.path.join(big_store_output_r, image_dir)
            small_output_dir = os.path.join(small_store_output_r, image_dir)

            blank_output = np.zeros((1, 3), dtype=np.float32)
            blank_label = np.ones((1,), dtype=np.int32)
            if (is_valid(faces_root , landmarks_root , image_path ,\
                            confidence_threshold = 0.9, area_low_threshold = 48*48, \
                             area_high_threshold = float('inf') )==True) or \
                    is_valid(faces_root, landmarks_root, image_path, \
                             confidence_threshold=0.9, area_low_threshold=12 * 12, \
                             area_high_threshold=48 * 48) == True:



                big_flag  = False
                if (is_valid(faces_root , landmarks_root , image_path ,\
                            confidence_threshold = 0.9, area_low_threshold = 48*48, \
                             area_high_threshold = float('inf') )==True):

                    assert (os.path.exists(big_output_dir))
                    big_output_file = os.path.join(big_output_dir, image_name + '.txt')
                    big_output = np.loadtxt(big_output_file, ndmin=2)

                    big_areas = get_area(landmarks_root_r,  image_path, \
                                                              .90, 48 * 48, float('inf'), nimg=10)
                    big_areas = np.array(big_areas)
                    big_areas = np.expand_dims(big_areas, axis=1)
                    big_flag = True

                small_flag = False

                if (is_valid(faces_root , landmarks_root , image_path ,\
                            confidence_threshold = 0.9, area_low_threshold = 12*12, \
                             area_high_threshold = 48*48 )==True):
                    assert (os.path.exists(small_output_dir))
                    small_output_file = os.path.join(small_output_dir, image_name + '.txt')
                    small_output = np.loadtxt(small_output_file, ndmin=2)
                    small_areas  = get_area(landmarks_root_r, image_path, \
                                                                  .90, 12 * 12, 48 * 48, nimg=10)
                    small_areas = np.array(small_areas)
                    small_areas = np.expand_dims(small_areas, axis=1)
                    small_flag = True

                if big_flag ==True  and small_flag == True:
                    output = np.concatenate( ( big_output , small_output))
                    areas =  np.concatenate( ( big_areas  , small_areas ))
                elif big_flag == True:
                    output = big_output
                    areas  = big_areas
                elif small_flag == True:
                    output = small_output
                    areas  = small_areas

                areas  = np.sqrt(areas)
                areas  = areas / np.sum(areas)
                output = output * areas
                # output = softmax( output , axis=1)

                    # print (output.shape)

                    # areas = get_area(landmarks_root_r, big_faces_root, \
                    #                  image_path, nimg=10)
                    # areas = np.array(areas)
                    # areas = np.expand_dims(areas, axis=1)
                    # output = output * areas

                # confidences = get_confidence(landmarks_root_fullpath, image_path, nimg=10)
                # confidences = np.array(confidences)
                # confidences = np.expand_dims(confidences , axis=1)
                # output = output*confidences


                # output = np.mean( output , axis= 0 , keepdims=True)
                output = np.sum(output, axis=0, keepdims=True)

                # images = process_image(faces_root , image_path)
                # images = images.cuda()
                # output = model( images )
                # output = output.mean(dim = 0 , keepdim = True)
                # output = output.max(dim=0, keepdim=True)[0]

                output_np = output
                outputs.append(output_np)
                filtered_labels.append(np.expand_dims(labels[id], axis=0))

            # delete tensors
            #     del output
            #     del images
            #     torch.cuda().empty_cache()

            else:
                #print ('No')
                miss_count = miss_count + 1
                outputs.append(blank_output)
                #filtered_labels.append(np.expand_dims(labels[id], axis=0))
                filtered_labels.append(blank_label)

        outputs = np.concatenate(outputs, axis=0)
        filtered_labels = np.concatenate(filtered_labels, axis=0)

        # print (store_output_r)

        run_output_faces = os.path.join(store_output_r, 'output_faces.txt')
        np.savetxt(run_output_faces, outputs)

        run_output_labels = os.path.join(store_output_r, 'output_labels.txt')
        np.savetxt(run_output_labels, filtered_labels)

        outputs_ensemble.append(outputs)
        filtered_labels_ensemble.append(filtered_labels)
        # print(get_accuracy(outputs, filtered_labels))

    outputs_ensemble = np.stack(outputs_ensemble , axis = 0)
    outputs_ensemble = np.mean( outputs_ensemble , axis=0 , keepdims=False)

    print (miss_count)

    filtered_labels = filtered_labels_ensemble[0]
    print( 'Face stream accuracy: ' + str(get_accuracy_ms(outputs_ensemble, filtered_labels , miss_count)))
    conf_mat = get_confusion_matrix(outputs_ensemble, filtered_labels)
    # print(get_confusion_matrix(outputs_ensemble, filtered_labels))
    # print(methods_util._buildstr(
    #     np.nan_to_num(100. * conf_mat / conf_mat.sum(axis=1, keepdims=True))))


    ensemble_output_faces = os.path.join( project_dir , store_outputs ,  'output_faces_ensemble.txt')
    np.savetxt(ensemble_output_faces, outputs)

    ensemble_output_labels = os.path.join(project_dir , store_outputs ,  'output_labels_ensemble.txt')
    np.savetxt(ensemble_output_labels ,filtered_labels)
    print ('End Combining face predicition')


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


def is_valid( root , landmarks_root , image_path , \
              confidence_threshold=.90, area_low_threshold=12 * 12, area_high_threshold=48 * 48):

    # processed_db = config_util.get_processed_db_dir()
    image_dir  = os.path.join(project_dir, root, image_path)[:-4]

    # print (image_dir)
    # assert( os.path.isdir( image_dir ))

    landmark_file = os.path.join(landmarks_root, image_path[:-4] , 'landmarks.txt')
    bounding_boxes_file = os.path.join(landmarks_root, image_path[:-4],  'bbox.txt')


    if not os.path.exists(landmark_file) or \
        not os.path.exists(bounding_boxes_file):
        return False

    bounding_boxes = np.loadtxt(bounding_boxes_file, ndmin=2)
    landmarks = np.loadtxt(landmark_file, ndmin=2)

    if len(bounding_boxes) == 0 or \
            len(landmarks) == 0:
        return False


    count = 0
    for id in range(len(landmarks)):

        area = get_area_bb(bounding_boxes[id])
        confidence = bounding_boxes[id][4]

        if area > area_low_threshold and area < area_high_threshold \
                and confidence > confidence_threshold:
            count += 1

    # print (count)
    if count :
        return True
    else:
        return False



def process_image( root , image_path , nimg = 10):

    processed_db = config_util.get_processed_db_dir()
    image_dir  = os.path.join(processed_db,root, image_path)[:-4]


    faces = os.listdir(image_dir)
    faces = filter( lambda x : x.endswith('.png') , faces )
    faces.sort()

    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    face_images = list()

    for id in range(len(faces)):
        if id >=nimg:
            break
        face = faces[id]
        face_fullpath = os.path.join( image_dir , face )
        image = Image.open(face_fullpath);
        image = image.convert('RGB')
        if transform:
            image = transform(image)

        face_images.append(image)

    face_images = torch.stack(face_images , dim = 0)
    return face_images



def get_area( root, image_path, confidence_threshold, area_low_threshold, area_high_threshold , nimg = 10):

    image_dir  = image_path[:-4]
    landmarks_dir = os.path.join(root , image_dir)
    # print (root, image_dir)

    # image_dir  = os.path.join(project_dir,faces_root, image_path)[:-4]

    ################## Loading Bounding Boxes ########

    bounding_boxes_file = os.path.join( landmarks_dir ,  'bbox.txt')
    landmark_file       = os.path.join( landmarks_dir , 'landmarks.txt')
    bounding_boxes = np.loadtxt(bounding_boxes_file,ndmin=2)
    landmarks      = np.loadtxt(landmark_file , ndmin=2)
    if len(bounding_boxes)==0 or \
            len(landmarks)==0:
        return None, 0

    ##################################################


    # faces = os.listdir(image_dir)
    # faces = filter(lambda x: x.endswith('.png'), faces)
    # faces.sort()


    bounding_boxes, landmarks = zip(*sorted(zip(bounding_boxes, landmarks), \
                                            key=lambda p: p[0][4], reverse=True))


    areas = list()
    # index = list()
    count = 0

    for i in range(len(landmarks)):

        if count >= nimg:
            break

        # face = faces[i]
        # face_temp = face.replace('.' , '_')
        # face_id =  int(face_temp.split('_')[1])
        face_id = i
        area = get_area_bb(bounding_boxes[face_id])
        confidence = bounding_boxes[face_id][4]

        if area > area_low_threshold and area < area_high_threshold \
                and confidence > confidence_threshold:
            count += 1
            areas.append( get_area_bb( bounding_boxes[face_id]))
            # index.append( int(face_id))




    return areas


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

def get_area_bb(  bb ):
    x = int(bb[0])
    y = int(bb[1])
    h = int((bb[3] - bb[1]))
    w = int((bb[2] - bb[0]))
    area = h * w
    return area

def get_confidence( root ,image_path, threshold = .94 , nimg = 10):

    image_dir  = image_path[:-4]
    landmarks_dir = os.path.join(root , image_dir)

    bounding_boxes_file = os.path.join( landmarks_dir ,  'bbox.txt')
    landmark_file = os.path.join( landmarks_dir , 'landmarks.txt')
    ################ Loading and filtering bbox
    bounding_boxes = np.loadtxt(bounding_boxes_file,ndmin=2)
    landmarks      = np.loadtxt(landmark_file , ndmin=2)
    if len(bounding_boxes)==0 or \
            len(landmarks)==0:
        return

    bounding_boxes , landmarks = zip(*sorted(zip(bounding_boxes,landmarks ),\
                                             key=lambda p:p[0][4] , reverse=True))
    thresholded = filter(lambda p:p[0][4] > threshold,
                                    zip(bounding_boxes, landmarks))
    if len(thresholded) == 0:
        return

    bounding_boxes , landmarks = zip(*thresholded)
    ######################################
    confidences  = list()
    biggest_height = 0
    for (i, bb) in enumerate(bounding_boxes):
        if i >= nimg:
            break
        confidences.append(bb[4])

    conf_sum = sum(confidences)
    confidences = [confidence / float(conf_sum) for confidence in confidences]
    # print (sum(areas))

    return confidences


def get_accuracy(outputs, labels):

    assert outputs.shape[0] == labels.shape[0]
    preds = np.argmax(outputs, axis=1)
    correct = (preds == labels).sum()
    total = labels.shape[0]
    acc = 1. * correct / total
    return acc

def get_accuracy_ms(outputs, labels , miss_count ):

    assert outputs.shape[0] == labels.shape[0]
    preds = np.argmax(outputs, axis=1)
    correct = (preds == labels).sum()
    total = labels.shape[0]
    acc = 1. * correct / (total - miss_count)
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
    args.add_argument("--start", default=1, type=int,
                        metavar='W', help='epoch-length (default: 2000)')
    args.add_argument("--end", default=10, type=int,
                        metavar='W', help='epoch-length (default: 2000)')

    return parser.parse_args()


if __name__=='__main__':
    main()