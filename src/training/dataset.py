from __future__ import print_function, division

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# import util.visualize as vs

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
import glob
import random
import pickle

from PIL import Image  # Replace by accimage when ready

########## Project Specific Common ##############
project_dir = os.getcwd()
sys.path.insert(0, project_dir)

# from util import config_util
# from util import methods_util
from src.util import multi_transforms


warnings.filterwarnings("ignore")

class ClassificationDataset(Dataset):

    def __init__(self, root, image_list, transform=None , is_grey = False):
        self.root = root;
        self.image_list = image_list;
        self.transform = transform;
        self._read_txt_file();
        self.is_grey = is_grey

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]);
        if not self.is_grey:
            image = image.convert('RGB')
        else:
            image = image.convert('L')

        label = self.labels[idx][0];
        if self.transform:
            image = self.transform(image)
        return image, label

    def _read_txt_file(self):

        self.img_paths = []
        self.labels = []

        file_path = self.image_list
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                items = line.split(' ')
                lbls = [int(s) for s in items[1:]]
                self.img_paths.append(os.path.join(self.root,
                                                   items[0]))
                self.labels.append(np.array(lbls))

        self.labels = np.array(self.labels);
        print (len(self.labels))
        return;


class ClassificationDatasetMulti(Dataset):

    def __init__(self, image_root, heatmap_root ,  image_list, transform=None , is_grey=False):

        self.image_root = image_root;
        self.image_list = image_list;

        self.heatmap_root = heatmap_root;

        self.transform = transform;
        self._read_txt_file();
        self.is_grey = is_grey

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Make sure the file exists
        assert (os.path.isfile(self.image_paths[idx]))
        assert (os.path.isfile(self.heatmap_paths[idx]))


        image   = Image.open(self.image_paths[idx]);
        image   = image.convert('RGB')
        heatmap = Image.open(self.heatmap_paths[idx]);
        heatmap = heatmap.convert('L')
        # if not self.is_grey:
        #     image = image.convert('RGB')
        #     heatmap = heatmap.convert('RGB')
        # else:
        #     image = image.convert('L')
        #     heatmap = heatmap.convert('L')

        label = self.labels[idx][0];
        image_list = [image , heatmap]

        # Don't want normalize to be applied on heatmap
        if self.transform:
            transforms = self.transform.transforms
            length  = len(transforms)
            for t in range(length - 1):
                image_list = transforms[t](image_list)

            image_list[0] = transforms[length -1](image_list[0])



        # Concat image accross channels !!!
        cat_img = torch.cat(image_list)
        return cat_img , label

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.image_paths = []
        self.heatmap_paths = []
        self.labels = []

        # file_path = os.path.join(self.root, self.image_list)
        file_path = self.image_list
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                lbls = [int(s) for s in items[1:]]
                self.image_paths.append(os.path.join(self.image_root,
                                                   items[0]))
                self.heatmap_paths.append(os.path.join(self.heatmap_root,
                                                   items[0]))
                self.labels.append(np.array(lbls))

        self.labels = np.array(self.labels);
        return;


class ClassificationDatasetHourglass(Dataset):

    def __init__(self, image_root, heatmap_root ,  image_list, transform=None , is_grey=False):

        self.image_root = image_root;
        self.image_list = image_list;

        self.heatmap_root = heatmap_root;

        self.transform = transform;
        self._read_txt_file();
        self.is_grey = is_grey

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Make sure the file exists
        assert (os.path.isfile(self.image_paths[idx]))
        assert (os.path.isfile(self.heatmap_paths[idx]))


        image   = Image.open(self.image_paths[idx]);
        image   = image.convert('RGB')
        heatmap = Image.open(self.heatmap_paths[idx]);
        heatmap = heatmap.convert('L')
        # if not self.is_grey:
        #     image = image.convert('RGB')
        #     heatmap = heatmap.convert('RGB')
        # else:
        #     image = image.convert('L')
        #     heatmap = heatmap.convert('L')

        label = self.labels[idx][0];
        image_list = [image , heatmap]

        # Don't want normalize to be applied on heatmap
        if self.transform:
            p_transforms = self.transform.transforms
            length  = len(p_transforms)
            for t in range(length - 2):
                image_list = p_transforms[t](image_list)

            # resize the heatmap before the toTensor call
            image_list[1] = transforms.Resize(64)( image_list[1])
            image_list = p_transforms[length-2](image_list)
            image_list[0] = p_transforms[length -1](image_list[0])



        image = image_list[0]
        # Copy the heatmap into appropriate channel
        heatmap = torch.zeros( [3 , 64 , 64 ] , dtype=torch.float32 )
        heatmap[label].copy_(image_list[1].view(64,64))
        # heatmap = torch.zeros()

        # Concat image accross channels !!!
        # cat_img = torch.cat(image_list)

        return image , heatmap , label

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.image_paths = []
        self.heatmap_paths = []
        self.labels = []

        # file_path = os.path.join(self.root, self.image_list)
        file_path = self.image_list
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                lbls = [int(s) for s in items[1:]]
                self.image_paths.append(os.path.join(self.image_root,
                                                   items[0]))
                self.heatmap_paths.append(os.path.join(self.heatmap_root,
                                                   items[0]))
                self.labels.append(np.array(lbls))

        self.labels = np.array(self.labels);
        return;

class FeatureExtraction(Dataset):

    def __init__(self, root  , image_list, transform=None , is_grey = False):
        self.root = root;
        self.image_list = image_list;
        self.transform = transform;
        self._read_txt_file();
        self.is_grey = is_grey

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]);
        if not self.is_grey:
            image = image.convert('RGB')
        else:
            image = image.convert('L')


        if self.transform:
            image = self.transform(image)

        return image, self.rel_img_paths[idx]

    def _read_txt_file(self):

        self.img_paths = []
        self.rel_img_paths = []


        file_path = self.image_list
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split('\n')
                self.img_paths.append(os.path.join(self.root,
                                                   items[0]))
                self.rel_img_paths.append( items[0] )

        return;

class LSTMDataset(Dataset):
    def __init__(self, root, image_list, transform=None ,
                 mxlen = 5):
        self.root = root;
        self.image_list = image_list;
        self.transform = transform;
        self.mxlen = mxlen
        self._read_txt_file();


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):


        label = self.labels[idx][0];
        feats = self.feature_paths[idx]

        feat_list = list()
        for i in range(len(feats)):
            feat = np.loadtxt(feats[i])
            feat_list.append(feat)

        feat_ndarray = np.array(feat_list)
        return feat_ndarray , label



    def _read_txt_file(self):

        self.img_paths = []
        self.feature_paths = []
        self.labels = []

        file_path = self.image_list
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                lbls = [int(s) for s in items[1:]]
                self.img_paths.append(os.path.join(self.root,
                                                   items[0]))
                self.labels.append(np.array(lbls))


        # Remove examples with no faces
        npath = []
        nlabels = []
        count = 0
        for idx in range(len(self.img_paths)):
            img_path = os.path.join(self.root ,
                                    self.img_paths[idx])
            feat_paths = os.path.join(img_path[:-4] , '*.txt')
            feats = glob.glob(feat_paths)

            if ( len(feats) ) != 0:
                npath.append(self.img_paths[idx])
                nlabels.append(self.labels[idx])
                count = count+1


        self.img_paths = npath
        self.labels = nlabels
        ####################################3#
        print(len(self.labels))

        # Set the labels
        self.labels = np.array(self.labels);

        for idx in range(len(self.img_paths)):
            img_path = os.path.join(self.root ,
                                    self.img_paths[idx])
            feat_paths = os.path.join(img_path[:-4] , '*.txt')
            # feat_paths = sorted(feat_paths)
            feats =  glob.glob(feat_paths)
            feats =  sorted(feats)

            if(len(feats)) >= self.mxlen:
                feats = feats[0:self.mxlen]
            elif( len(feats)!=0):
                repeats = int(self.mxlen / len(feats))
                feats = feats*(repeats+1)
                feats = feats[0:self.mxlen]


            self.feature_paths.append(feats)




        return;

class LSTMDatasetEmoAV(Dataset):
    def __init__(self, image_list, feature_list , transform=None ,
                 mxlen = 20):

        self.image_list   = image_list
        self.feature_list = feature_list
        self.transform = transform;
        self.mxlen = mxlen
        self._read_txt_file();


    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):

        video_id = self.video_ids[idx]

        label = self.labels[video_id];
        feats = self.feature_dict[video_id]
        ln= len(feats)

        clen = min( self.mxlen , ln )
        random_feature_ids = random.sample( range(ln) , clen )
        random_feature_ids.sort()


        feat_list = list()
        for i in random_feature_ids:
            feat = feats[ i ]
            feat_list.append( feat )

        for i in range(self.mxlen - ln):
            feat = np.zeros((1024,))
            feat_list.append(feat)

        feat_ndarray = np.array(feat_list)
        return feat_ndarray , label ,clen



    def _read_txt_file(self):

        self.feature_dict = dict()
        self.labels = dict()
        self.video_ids = list()

        labels_local   = np.loadtxt(self.image_list, dtype='str')
        features_local = np.loadtxt(self.feature_list, dtype='float')

        self.feature_dict = dict()
        for i in range(len(labels_local)):
            frame_path = labels_local[i][0]
            video_id = frame_path.split('/')[-2]

            if video_id not in self.feature_dict:
                self.feature_dict[video_id] = list()
                self.labels[video_id] = int(labels_local[i][1])
                self.video_ids.append(video_id)

            self.feature_dict[video_id].append(features_local[i])
        return;


class LSTMDatasetEmoAVPkl(Dataset):
    def __init__(self, pkl , transform=None ,
                 mxlen = 20):

        self.pkl = pkl
        self.transform = transform;
        self.mxlen = mxlen
        self._read_txt_file();


    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):

        video_id = self.video_ids[idx]

        label = self.labels[video_id];
        feats = self.feature_dict[video_id]
        ln= len(feats)

        clen = min( self.mxlen , ln )
        random_feature_ids = random.sample( range(ln) , clen )
        random_feature_ids.sort()

        # print (clen)

        feat_list = list()
        for i in random_feature_ids:
            feat = feats[ i ]
            feat_list.append( feat )

        for i in range(self.mxlen - ln):
            feat = np.zeros((4096,))
            feat_list.append(feat)

        feat_ndarray = np.array(feat_list).astype(np.float32)
        return feat_ndarray , label ,clen



    def _read_txt_file(self):

        self.feature_dict = dict()
        self.labels = dict()
        self.video_ids = list()

        fileObject_r = open(self.pkl, 'r')
        self.feature_dict, self.labels, self.video_ids = pickle.load(fileObject_r)
        fileObject_r.close()

        return;

class LRCNDataset(Dataset):

    def __init__(self, root, image_list, transform=None ,
                 mxlen = 12):

        self.root = root;
        self.image_list = image_list;
        self.transform = transform;
        self.mxlen = mxlen
        self._read_txt_file();


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx][0];
        images = self.image_paths[idx]

        image_list = list()
        ln  = len(images)
        for image_path in images:
            image = Image.open(image_path);
            image = image.convert('RGB')
            image_list.append( image )


        if self.transform:
            transforms_local = self.transform.transforms
            length  = len(transforms_local)
            for t in range(length):
                image_list = transforms_local[t](image_list)

        zero_padding = torch.zeros( image_list[0].shape )

        for _ in range( self.mxlen - ln ):
            image_list.append( zero_padding )


        # stack_img = torch.stack( image_list )
        stack_img = torch.cat( image_list )
        return stack_img , label , ln



    def _read_txt_file(self):

        self.img_dirs = []
        self.image_paths = []
        self.labels = []

        file_path = self.image_list
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                lbls = [int(s) for s in items[1:]]
                self.img_dirs.append(os.path.join(self.root,
                                                   items[0]))
                self.labels.append(np.array(lbls))


        print(len(self.labels))

        # Set the labels
        self.labels = np.array(self.labels);

        for idx in range(len(self.img_dirs)):
            img_dir = os.path.join(self.root ,
                                    self.img_dirs[idx])

            image_dir_regex = os.path.join(img_dir , '*.png')
            # feat_paths = sorted(feat_paths)
            images =  glob.glob(image_dir_regex)
            images =  sorted(images)

            if(len(images)) >= self.mxlen:
                images = images[0:self.mxlen]



            self.image_paths.append(images)

        return;

class FeaturCombinationDataset(Dataset):
    def __init__(self, root, image_list, transform=None ):

        self.root = root;
        self.image_list = image_list;
        self.transform = transform;
        self._read_txt_file();


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        label = self.labels[idx][0];
        feats = self.feature_paths[idx]

        feat_avg = np.zeros(shape=(4096,) , dtype=float )

        feat_list = list()
        for i in range(len(feats)):
            feat = np.loadtxt(feats[i])
            feat_avg = feat_avg + feat

        feat_avg  = feat_avg / len(feats)
        return feat_avg , label



    def _read_txt_file(self):

        self.img_paths = []
        self.feature_paths = []
        self.labels = []

        file_path = self.image_list
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                lbls = [int(s) for s in items[1:]]
                self.img_paths.append(os.path.join(self.root,
                                                   items[0]))
                self.labels.append(np.array(lbls))





        # Set the labels
        self.labels = np.array(self.labels);

        for idx in range(len(self.img_paths)):
            img_path = os.path.join(self.root ,
                                    self.img_paths[idx])
            feat_paths = os.path.join(img_path[:-4] , '*.txt')
            # feat_paths = sorted(feat_paths)
            feats =  glob.glob(feat_paths)
            feats =  sorted(feats)


            self.feature_paths.append(feats)




        return;

def imshow(inp, title=None,
           mean = [0.485, 0.456, 0.406] ,
           std  = [0.229, 0.224, 0.225]):
    """Imshow for Tensor."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array(mean)
    std = np.array(std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # Here pause for 5 seconds
    plt.pause(5.0)  # pause a bit so that plots are updated


def main():
    print ("In main")
    # image_list = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018_av/vgg_features/AFEW2018_Val.txt'
    # feature_list = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018_av/vgg_features/result_score_val_fc5_1_20000'

    # dataset = LSTMDatasetEmoAV(
    #                             image_list   = image_list ,
    #                             feature_list = feature_list
    #                           );
    pkl = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018_av/vgg_features/test_n.pickle'
    dataset = LSTMDatasetEmoAVPkl( pkl=pkl

                              );
    # data = dataset[5]
    # print (data[0].shape)

    # Balancing the classes !!!!!!
    # prob = np.zeros(7)
    # for i in range(len(dataset)):
    #     cur_class = dataset.labels[dataset.video_ids[i]]
    #     prob[cur_class]+=1
    # prob = 1.0 / prob
    #
    #
    # reciprocal_weights = np.zeros(len(dataset))
    # epoch_length = 400
    # for i in range(len(dataset)):
    #     label = dataset.labels[dataset.video_ids[i]]
    #     reciprocal_weights[i] = prob[label]
    # weights  = torch.from_numpy(reciprocal_weights)
    #
    # weighted_sampler = sampler.WeightedRandomSampler(weights , epoch_length)
    # loader = DataLoader(dataset, batch_size=10,
    #                     sampler=weighted_sampler)

    # loader = DataLoader(dataset, batch_size=20)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)


    for i_batch, data in enumerate(loader):
        features , labels , ln = data

        # images = images.squeeze(dim = 0)
        # # print (images.shape)
        # images = images.view( -1 , 3, 224,224)
        # #
        # #
        # out = utils.make_grid(images , nrow=4)
        # imshow(out)

        # vs.imshow(out , mean = [0.524462] , std = [0.285962])
        # vs.imshow(out, mean=[0.524462, 0.524462, 0.524462], std=[0.285962, 0.285962, 0.285962])
        print (features.shape)
        print (type(features))
        break
    print ("End Main")



if __name__ == '__main__':
    main()

def LRCNDatasetTest():
    root           = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/cropped_images/group_based_faces_alligned_94'
    image_list     = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/partition/faces_alligned_94_lstm/val.txt'

    dataset = LRCNDataset(root,
                                image_list,
                                transform=transforms.Compose([
                                    multi_transforms.ResizeMulti(256),
                                    multi_transforms.RandomRotationMulti(3),
                                    # transforms.RandomCrop(224),
                                    multi_transforms.RandomResizedCropMulti(224, scale=(0.74, 0.78),
                                                                 ratio=(1.0, 1.0)),
                                    multi_transforms.RandomHorizontalFlipMulti(),
                                    multi_transforms.ToTensorMulti(),
                                    multi_transforms.NormalizeMulti([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                                );

    # data = dataset[5]
    # print (data[0].shape)

    # Balancing the classes !!!!!!
    # prob = np.zeros(7)
    # for i in range(len(dataset)):
    #     cur_class = dataset[i][1]
    #     prob[cur_class]+=1
    # prob = 1.0 / prob
    #
    #
    # reciprocal_weights = np.zeros(len(dataset))
    # epoch_length = 2000
    # for i in range(len(dataset)):
    #     _, label = dataset[i]
    #     reciprocal_weights[i] = prob[label]
    # weights  = torch.from_numpy(reciprocal_weights)
    #
    #
    # weighted_sampler = sampler.WeightedRandomSampler(weights , epoch_length)
    # loader = DataLoader(dataset, batch_size=10,
    #                     sampler=weighted_sampler)

    # loader = DataLoader(dataset, batch_size=20)

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i_batch, data in enumerate(loader):
        images , labels , ln = data
        # images = images.squeeze(dim = 0)
        # print (images.shape)
        images = images.view( -1 , 3, 224,224)
        #
        #
        out = utils.make_grid(images , nrow=4)
        imshow(out)

        # vs.imshow(out , mean = [0.524462] , std = [0.285962])
        # vs.imshow(out, mean=[0.524462, 0.524462, 0.524462], std=[0.285962, 0.285962, 0.285962])
        break
    print ("End Main")

def ClassificationDatasetHourglassTest():
    root           = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/cropped_images/global_cropped_264'
    heatmap_root = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/cropped_images/group_based_heatmaps_264_94'
    image_list     = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/partition/global_heatmap/train.txt'

    dataset = ClassificationDatasetHourglass(root, heatmap_root,
                                image_list,
                                transform=transforms.Compose([
                                    multi_transforms.ResizeMulti(256),
                                    multi_transforms.RandomRotationMulti(3),
                                    # transforms.RandomCrop(224),
                                    multi_transforms.RandomResizedCropMulti(224, scale=(0.74, 0.78),
                                                                 ratio=(1.0, 1.0)),
                                    multi_transforms.RandomHorizontalFlipMulti(),
                                    multi_transforms.ToTensorMulti(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                                );

    # Balancing the classes !!!!!!
    # prob = np.zeros(7)
    # for i in range(len(dataset)):
    #     cur_class = dataset[i][1]
    #     prob[cur_class]+=1
    # prob = 1.0 / prob
    #
    #
    # reciprocal_weights = np.zeros(len(dataset))
    # epoch_length = 2000
    # for i in range(len(dataset)):
    #     _, label = dataset[i]
    #     reciprocal_weights[i] = prob[label]
    # weights  = torch.from_numpy(reciprocal_weights)
    #
    #
    # weighted_sampler = sampler.WeightedRandomSampler(weights , epoch_length)
    # loader = DataLoader(dataset, batch_size=10,
    #                     sampler=weighted_sampler)

    # loader = DataLoader(dataset, batch_size=20)

    loader = DataLoader(dataset, batch_size=20, shuffle=True)
    for i_batch, data in enumerate(loader):
        images , heatmap, labels = data
        print(labels)

        print(heatmap.max())
        out = utils.make_grid(heatmap , nrow=4)
        imshow(out)

        # vs.imshow(out , mean = [0.524462] , std = [0.285962])
        # vs.imshow(out, mean=[0.524462, 0.524462, 0.524462], std=[0.285962, 0.285962, 0.285962])
        break

def LSTMDatasetTest():

    print ("In main")
    root  = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/extracted_features/group_based_faces_vgg_16_all_emo'
    image_list = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/partition/global/train.txt'
    dataset = LSTMDataset(root, image_list);

    x = dataset[3]

    loader = DataLoader(dataset, batch_size=10,
                        shuffle=True)

    for i_batch, data in enumerate(loader):
        images , labels = data
        print (images.shape)
    print ("End Main")


def FeatureExtractionTest():
    image_list = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/partition/faces/all_faces.txt'
    root = '/media/shehabk/E_DRIVE/processed_db/emotiw_2018/group_based_faces_alligned'
    dataset = FeatureExtraction(root,
                               image_list,
                               transform=transforms.Compose([
                                   transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ]));


    for i in range(5):
        print (dataset[i][1])


def ClassificationDatasetTest():
    root       = '/media/shehabk/E_DRIVE/processed_db/expw/cropped_images/cropped_alligned_orig_256'
    image_list = '/media/shehabk/E_DRIVE/processed_db/expw/partition/seven_neutral_alligned_orig/train.txt'
    dataset = ClassificationDataset(root,
                                       image_list,
                                       transform=transforms.Compose([
                                           transforms.RandomRotation(3),
                                           transforms.Resize((118, 100)),
                                           transforms.RandomCrop((112, 96)),
                                           # transforms.RandomResizedCrop(224, scale=(0.74, 0.78),
                                           #                              ratio=(1.0, 1.0)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ]));

    # Balancing the classes !!!!!!
    prob = np.zeros(3)
    for i in range(len(dataset)):
        cur_class = dataset[i][1]
        prob[cur_class]+=1
    prob = 1.0 / prob


    reciprocal_weights = np.zeros(len(dataset))
    epoch_length = 2000
    for i in range(len(dataset)):
        _, label = dataset[i]
        reciprocal_weights[i] = prob[label]
    weights  = torch.from_numpy(reciprocal_weights)


    weighted_sampler = sampler.WeightedRandomSampler(weights , epoch_length)
    loader = DataLoader(dataset, batch_size=10,
                        sampler=weighted_sampler)

    # loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for i_batch, data in enumerate(loader):
        images , labels = data
        print(labels)
        out = utils.make_grid(images, nrow=1)
        imshow(out)

        # vs.imshow(out , mean = [0.524462] , std = [0.285962])
        # vs.imshow(out, mean=[0.524462, 0.524462, 0.524462], std=[0.285962, 0.285962, 0.285962])
        break