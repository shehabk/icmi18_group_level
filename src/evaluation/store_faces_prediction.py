from __future__ import print_function
# torch imports
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.autograd import Variable
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
#####################################


########## Project Specific Common ##############
project_dir = os.getcwd()
sys.path.insert(0, project_dir)
################################################

# from util import config_util
# from util import methods_util
# from util import multi_transforms
# import datasets.emotiw as ds
import torchvision.models as torchmodels
# import cmodels.zibonet as zibonet
##################################################


def main():
    print ('Start storing face evaluation')

    # arch  = 'resnet_i48_18'
    # arch  = 'resnet_i48_cl_18'
    args = parse_arguments()
    arch = args.arch
    allignment = args.allignment
    trained_on = args.trained_on
    test_data  = args.test_data

    if arch == 'resnet_i24_34':
        confidence_threshold = 0.90
        area_low_threshold = 12 * 12
        area_high_threshold = 48 * 48
    elif arch == 'resnet_i48_18':
        confidence_threshold = 0.90
        area_low_threshold = 48 * 48
        area_high_threshold = float('inf')

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
    store_outputs =    'data/predictions/face/%s/%s/%s/%s'%(allignment,
                                                            trained_on,
                                                            test_data,
                                                            arch+arch_pad)

    trained_model_root = 'models/faces/%s/%s'%(trained_on, arch)
    landmarks_root = 'data/landmarks'



    for i in range(start,end+1):
        epad  = 'run'+ str(i)
        print (epad)
        if choose_last_model == True:
            model_name = 'checkpoint.pth.tar'
        else:
            model_name = 'model_best.pth.tar'

        store_output_r  = os.path.join( project_dir , store_outputs , epad)
        trained_model_r = os.path.join( project_dir ,trained_model_root , epad , 'models/%s'%(model_name))
        image_list_r    = os.path.join(  project_dir , image_list)


        if not os.path.exists(store_output_r):
            os.makedirs(store_output_r)

        image_paths , labels = read_imagelist(image_list_r)
        model = get_model(trained_model_r , arch = arch)
        model.eval()


        for id in range( len(image_paths)):
            image_path = image_paths[id]

            tokens = image_path.split('/')
            image_dir = '/'.join(tokens[:-1])
            image_name = tokens[-1][:-4]
            output_dir = os.path.join(store_output_r , image_dir )


            if (is_valid(faces_root , landmarks_root , image_path ,\
                    confidence_threshold, area_low_threshold, area_high_threshold )==True):

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_file = os.path.join( output_dir , image_name + '.txt')

                images = process_image(faces_root , landmarks_root,  image_path , 10 ,  arch , \
                                       confidence_threshold, area_low_threshold, area_high_threshold)
                images = images.cuda()
                output = model( images )


                # output = output.mean(dim = 0 , keepdim = True)
                # output = output.max(dim=0, keepdim=True)[0]

                output_np = output.data.cpu().numpy()
                # print (output_file)
                np.savetxt( output_file , output_np )

    print ('End storing face evaluation')


def get_model( model_path , arch = 'resnet_i48_18'):

    num_classes = 3

    if arch == 'resnet_i48_18':
        model = torchmodels.resnet18()
        model = Renset_i48(model, num_classes=num_classes)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    elif arch == 'resnet_i24_34':
        model = torchmodels.resnet34()
        model = Renset_i24(model, num_classes=num_classes , input_size= 24)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

    return model.cuda()


def get_area_bb(  bb ):
    x = int(bb[0])
    y = int(bb[1])
    h = int((bb[3] - bb[1]))
    w = int((bb[2] - bb[0]))
    area = h * w
    return area

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



def process_image( root , landmarks_root , image_path , nimg = 10 , arch = 'resnet_i48_18', \
                   confidence_threshold = .90, area_low_threshold = 12*12, area_high_threshold = 48*48):


    image_dir  = os.path.join(project_dir, root, image_path)[:-4]


    faces = os.listdir(image_dir)
    faces = filter( lambda x : x.endswith('.png') , faces )
    faces.sort()

    landmark_file = os.path.join(landmarks_root, image_path[:-4] , 'landmarks.txt')
    bounding_boxes_file = os.path.join(landmarks_root, image_path[:-4],  'bbox.txt')

    # print(landmark_file)
    assert os.path.exists(landmark_file)
    assert os.path.exists(bounding_boxes_file)

    bounding_boxes = np.loadtxt(bounding_boxes_file, ndmin=2)
    landmarks = np.loadtxt(landmark_file, ndmin=2)

    if len(bounding_boxes) == 0 or \
            len(landmarks) == 0:
        return

    bounding_boxes, landmarks, faces = zip(*sorted(zip(bounding_boxes, landmarks , faces), \
                                            key=lambda p: p[0][4], reverse=True))


    if arch == 'resnet_i48_18':

        transform = transforms.Compose([
            transforms.Resize(60),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.28,))
        ])

        face_images = list()
        count = 0

        for id in range(len(faces)):

            if count >= nimg:
                break

            area = get_area_bb(bounding_boxes[id])
            confidence = bounding_boxes[id][4]

            if area > area_low_threshold and area < area_high_threshold \
                    and confidence > confidence_threshold:
                count += 1
                face = faces[id]
                face_fullpath = os.path.join(image_dir, face)
                image = Image.open(face_fullpath);
                image = image.convert('RGB')
                if transform:
                    image = transform(image)
                face_images.append(image)


    elif arch == 'resnet_i24_34':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(24),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        face_images = list()
        count = 0
        for id in range(len(faces)):

            if count >= nimg:
                break

            area = get_area_bb(bounding_boxes[id])
            confidence = bounding_boxes[id][4]

            if area > area_low_threshold and area < area_high_threshold \
                    and confidence > confidence_threshold:
                count += 1
                face = faces[id]
                face_fullpath = os.path.join(image_dir, face)
                image = Image.open(face_fullpath);
                image = image.convert('RGB')
                if transform:
                    image = transform(image)
                face_images.append(image)


    face_images = torch.stack(face_images , dim = 0)
    return face_images




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


class Renset_i48(nn.Module):
    def __init__(self, original_model , num_classes = 3 , input_size = 48 ):
        super(Renset_i48, self).__init__()
        original_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        childrens = list(original_model.children())
        self.feat = nn.Sequential(*childrens[:-2])

        cw  = self.get_channel_width()
        self.avgpool = nn.AvgPool2d(cw, stride=1)

        feat_size = self.get_feat_size()
        self.fc      = nn.Linear( feat_size , num_classes )

        # self.get_feat_size()

    def forward(self, x ):
        x = self.feat(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out   = self.fc(x)
        return out

    def get_feat_size(self ):
        dummy_input = Variable(torch.randn(1, 3, 48, 48))
        feat = self.feat(dummy_input)
        feat = self.avgpool(feat)
        shape = feat.shape
        return int( shape[1]*shape[2]*shape[3])


    def get_channel_width(self):
        dummy_input = Variable(torch.randn(1, 3, 48, 48))
        feat = self.feat(dummy_input)
        shape = feat.shape
        return int(shape[3])



class Renset_i24(nn.Module):
    def __init__(self, original_model, num_classes=3, input_size = 24):
        super(Renset_i24, self).__init__()
        original_model.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        childrens = list(original_model.children())
        self.input_size = input_size
        self.feat = nn.Sequential(*childrens[:-2])

        cw = self.get_channel_width()
        self.avgpool = nn.AvgPool2d(cw, stride=1)

        feat_size = self.get_feat_size()
        self.fc = nn.Linear(feat_size, num_classes)

        # self.get_feat_size()

    def forward(self, x):
        x = self.feat(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def get_feat_size(self):
        dummy_input = Variable(torch.randn(1, 3, self.input_size, self.input_size))
        feat = self.feat(dummy_input)
        feat = self.avgpool(feat)
        shape = feat.shape
        return int(shape[1] * shape[2] * shape[3])

    def get_channel_width(self):
        dummy_input = Variable(torch.randn(1, 3, self.input_size, self.input_size))
        feat = self.feat(dummy_input)
        shape = feat.shape
        return int(shape[3])


def parse_arguments():
    parser          = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.add_argument_group('args', 'experiment specific arguments')

    args.add_argument("--arch", metavar='N', default='resnet_i24_34',
                         help='architecture to use')
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