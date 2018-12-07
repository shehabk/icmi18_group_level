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
from multiprocessing import Pool
###################################


########## Project Specific Common ##############
project_dir = os.getcwd()
sys.path.insert(0, project_dir)
################################################


# from util import config_util
from src.util import methods_util
import dataset as ds
import torchvision.models as torchmodels
##################################################



def main(args):
    # print (vars(args))

    experiment_name = args.experiment_name
    exp_root  = project_dir
    processed_db_dir = project_dir



    exp_root  = os.path.join(exp_root, experiment_name)
    arch = args.architecture




    if (args.mode == 'train' or args.mode == 'sc'):
        root = os.path.join(processed_db_dir,args.image_root)

        train_file = os.path.join( processed_db_dir,  args.train_list)
        test_file  = os.path.join( processed_db_dir,   args.test_list)
        val_file   = os.path.join( processed_db_dir,  args.valid_list)


        # print (train_file)
        assert os.path.isfile(train_file)
        assert os.path.isfile(test_file )
        assert os.path.isfile(val_file  )

        ########## Additional Arguments ############
        args.exp_dir = os.path.join(exp_root, arch , args.experiment_padding)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)

        args.dataloader_train = get_dataloader_train(args, root , train_file)
        args.dataloader_test  = get_dataloader_test (args, root , val_file)
        args.dataloader_val   = get_dataloader_test (args, root , val_file)
        args.log_prefix       = experiment_name + '_' + arch + '_' + args.experiment_padding
        ##############################################

        exp = ClassificationTrainer(args)
        if ( args.mode == 'train'):
            exp.run_train()
        if ( args.mode == 'sc'):
            exp.run_write_score()

    if (args.mode == 'cp'):

        ########## Additional Arguments ############
        args.exp_dir = os.path.join(exp_root, arch, args.experiment_padding)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        ##############################################

        # methods_util.copy_to_shehabk(args.exp_dir)




    if (args.mode =='acc'):

        acc = []



        ########## Additional Arguments ############
        args.exp_dir = os.path.join(exp_root, arch, args.experiment_padding)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        ##############################################

        reuslts_dir = os.path.join(args.exp_dir , 'results')
        score_file_path = os.path.join(reuslts_dir,
                                       'score.txt')
        label_file_path = os.path.join(reuslts_dir,
                                       'label.txt')

        outputs = np.loadtxt(score_file_path)
        labels = np.loadtxt(label_file_path)


        exp = ClassificationTrainer(args)
        acc.append(exp.get_accuracy(outputs,labels,args.which_way))


        print ("Average: %.4f"% (sum(acc)/float(len(acc))))


    if ( args.mode == 'cf'):

        cfs = []

        ########## Additional Arguments ############
        args.exp_dir = os.path.join(exp_root, arch, args.experiment_padding)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
        ##############################################

        reuslts_dir = os.path.join(args.exp_dir , 'results')
        score_file_path = os.path.join(reuslts_dir,
                                       'score.txt')
        label_file_path = os.path.join(reuslts_dir,
                                       'label.txt')

        outputs = np.loadtxt(score_file_path)
        labels  = np.loadtxt(label_file_path)

        exp = ClassificationTrainer(args)
        cfs.append(exp.get_confusion_matrix(outputs, labels, num_classes=args.num_classes))

        num_classes = args.num_classes
        cf_sum = np.zeros((num_classes, num_classes))
        cf_sum+=cfs[0] #only one set!!!


        print ("Average:")
        print(methods_util._buildstr(100*
            np.nan_to_num(1. * cf_sum / cf_sum.sum(axis=1, keepdims=True))))




def get_dataloader_train(args , root, image_list):
    # kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    kwargs = {'num_workers': args.num_workers  , 'pin_memory': args.pin_memory}


    dataset = ds.ClassificationDataset(root,
                                image_list,
                                transform=transforms.Compose([
                                    transforms.RandomRotation(3),
                                    # transforms.RandomCrop(224),
                                    transforms.RandomResizedCrop(224, scale=(0.74, 0.90),
                                                                 ratio=(0.9, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ]),
                                );



    # # Balancing the classes !!!!!!
    # prob = np.zeros(args.num_classes)
    # for i in range(len(dataset)):
    #     cur_class = dataset[i][1]
    #     prob[cur_class]+=1
    # prob = 1.0 / prob
    #
    #
    # reciprocal_weights = np.zeros(len(dataset))
    # epoch_length = args.epoch_length
    # for i in range(len(dataset)):
    #     _, label = dataset[i]
    #     reciprocal_weights[i] = prob[label]
    # weights  = torch.from_numpy(reciprocal_weights)
    #
    #
    # weighted_sampler = sampler.WeightedRandomSampler(weights , epoch_length)
    # loader = DataLoader(dataset, batch_size=args.batch_size,
    #                     sampler=weighted_sampler, **kwargs)


    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return loader


def get_dataloader_test(args, root, image_list):

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory}


    dataset = ds.ClassificationDataset(root,
                                  image_list,
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                  ]));



    loader = DataLoader(dataset, batch_size=args.batch_size//3,
                        shuffle=False, **kwargs)

    return loader


class ClassificationTrainer(object):
    def __init__ ( self , args ):
        self.args = args
        self.best_acc = 0
        self.pool = Pool(processes=1)
        self.global_step = 0

        if (self.args.tb_log == True):
            self.tblogger = self.get_tblogger()

        if (self.args.mode == 'train' or
            self.args.mode == 'test' or
            self.args.mode ==  'sc'):

            self.model = self.get_model()
            self.optimizer = self.get_optimizer()
            self.criterion = self.get_criterion()

            self.dataloader_train = self.args.dataloader_train
            self.dataloader_test  = self.args.dataloader_test
            self.dataloader_val   = self.args.dataloader_val


    def run_train(self):

        ################ Sreen + Log ########
        log_file_path = os.path.join(self.args.exp_dir, 'log.txt')

        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path))

        if self.args.resume:
            log_file = open(log_file_path, 'a+')
        else:
            log_file = open(log_file_path, 'w+')

        sys.stdout = methods_util.Tee(log_file, sys.stdout)
        assert os.path.isfile(log_file_path)
        ########################################

        ################ Parameter Writing  ########
        args_file_path = os.path.join(self.args.exp_dir, 'args.txt')

        if not os.path.exists(os.path.dirname(args_file_path)):
            os.makedirs(os.path.dirname(args_file_path))


        methods_util.save_params(args_file_path,self.args)
        assert os.path.isfile(log_file_path)
        #############################################


        if self.args.resume:
            self.load_checkpoint(self.model, self.optimizer)

        for epoch in range(self.args.start_epoch, self.args.epochs + 1):
            self.adjust_learning_rate(self.optimizer, epoch)
            trainloss , trainacc = self.train(self.dataloader_train , self.model, self.optimizer, self.criterion, epoch)
            validloss , validacc = self.evaluate(self.dataloader_val, self.model, self.criterion)
            val_acc = validacc.avg
            is_best = val_acc >= self.best_acc
            self.best_acc = max(val_acc, self.best_acc)
            # print (set_name, val_acc,best_acc)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step
            }, is_best)

            if (self.args.tb_log == True):
                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'train_loss': trainloss.avg,
                    'train_acc' : trainacc.avg ,
                    'valid_loss': validloss.avg,
                    'valid_acc' : validacc.avg
                }

                for tag, value in info.items():
                    self.tblogger.scalar_summary(tag, value, epoch + 1)


            print('TRAIN SET: AVERAGE LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)'.format(
                trainloss.avg, int(trainacc.sum), trainacc.count,
                100. * trainacc.avg))
            print('TEST SET: AVERAGE LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)'.format(
                validloss.avg, int(validacc.sum), validacc.count,
                100. * validacc.avg))

        # if self.args.copy_back:
        #     self.pool.apply_async(methods_util.copy_to_shehabk ,[self.args.exp_dir])





    def run_write_score( self ):

        self.load_checkpoint(self.model, self.optimizer, True)
        self.write_score(self.dataloader_test, self.model )
        # if self.args.copy_back:
        #     self.pool.apply_async(methods_util.copy_to_shehabk ,[self.args.exp_dir])

    ######################### Standard Train Test ##########################
    def train(self , train_loader, model, optimizer, criterion, epoch ):
        # Change model in training mode

        losses     = AverageMeter()
        accuracies = AverageMeter()

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data , volatile = True), Variable(target, volatile= True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            # optimizer.step()

            # Gradient Clipping If necessary
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)
            optimizer.step()

            self.global_step+=1

            ############# Update Loss ###########
            losses.update(loss.data[0] , target.size(0))
            ############# Update Accuracy #######
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc =   float(correct) / target.size(0)
            accuracies.update(acc , target.size(0))

            if batch_idx % self.args.log_interval == 0:
                print('Train {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.args.log_prefix, epoch, batch_idx * len(data), len(train_loader.dataset),  # do train_loader.sampler
                                     100. * batch_idx / len(train_loader), loss.data[0]))

                if (self.args.tb_log == True):
                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'loss': loss.data[0],
                        # 'accuracy': accuracy.data[0]
                    }

                    for tag, value in info.items():
                        self.tblogger.scalar_summary(tag, value, self.global_step + 1)

                    # (2) Log values and gradients of the parameters (histogram)
                    for tag, value in model.named_parameters():

                        if value.grad is None:
                            continue

                        tag = tag.replace('.', '/')
                        self.tblogger.histo_summary(tag, value.data.cpu().numpy(), self.global_step + 1)
                        self.tblogger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), self.global_step + 1)

                    # (3) Log the images
                    # info = {
                    #     'images': to_np(images.view(-1, 28, 28)[:10])
                    # }
                    #
                    # for tag, images in info.items():
                    #     logger.image_summary(tag, images, step + 1)

        return losses , accuracies

    def evaluate(self, test_loader, model, criterion):
        # change model in eval mode (no Dropout like random events)

        losses     = AverageMeter()
        accuracies = AverageMeter()


        model.eval()

        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss   = criterion(output, target)  # sum up batch loss

            # output = F.softmax(output)

            ############# Update Loss ###########
            losses.update(loss.data[0] , target.size(0))
            ############# Update Accuracy #######
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            acc =   float(correct) / target.size(0)
            accuracies.update(acc , target.size(0))

        # print (get_accuracy(outputs, labels , 0 ))
        # print (methods_util._buildstr(get_confusion_matrix(outputs, labels)))

        # test_loss /= len(test_loader.dataset)
        # print('TEST SET: AVERAGE LOSS: {:.4f}, ACCURACY: {}/{} ({:.0f}%)'.format(
        #     losses.avg, accuracies.sum, accuracies.count,
        #     100. * accuracies.avg ))


        return losses , accuracies

    def write_score(self, test_loader, model, set_name=''):

        ################ Create Results Directory  #############
        results_dir = os.path.join(self.args.exp_dir, "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        ########################################################

        score_file_path = os.path.join( results_dir, 'score.txt')
        label_file_path = os.path.join( results_dir, 'label.txt')


        #############################################

        # change model in eval mode (no Dropout like random events)
        model.eval()

        outputs = []
        labels = []

        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target,volatile = True)
            output = model(data)

            # output = F.softmax(output)
            outputs.append(output.data.cpu().numpy())
            labels.append(target.data.cpu().numpy())

        outputs = np.concatenate(outputs, axis=0)
        labels = np.concatenate(labels, axis=0)

        np.savetxt(score_file_path, outputs)
        np.savetxt(label_file_path, labels)

    ############################## Utility Methods ###########################


    def get_accuracy(self , outputs, labels , which_way = 0 ):

        ##### expects already concatenated arrays
        # outputs = np.concatenate(outputs, axis = 0)
        # labels  = np.concatenate(labels , axis = 0)

        assert outputs.shape[0] == labels.shape[0]

        preds = np.argmax(outputs, axis=1)
        correct = (preds == labels).sum()
        total = labels.shape[0]
        acc = 1. * correct / total

        return acc




    def get_confusion_matrix(self, outputs, labels , which_way=0,
                             num_frames=3, num_classes=7):

        # outputs = np.concatenate(outputs, axis = 0)
        # labels  = np.concatenate(labels , axis = 0)

        assert outputs.shape[0] == labels.shape[0]
        if which_way!=0:
            assert labels.shape[0] % num_frames == 0

        conf_mat = np.zeros((num_classes, num_classes))

        if which_way == 1:
            outputs = np.reshape(outputs, (outputs.shape[0] / num_frames, num_frames, -1))
            outputs = np.sum(outputs, axis=1)
            labels = labels[::num_frames]

        if which_way == 2:
            outputs = np.reshape(outputs, (outputs.shape[0] / num_frames, num_frames, -1))
            outputs = np.max(outputs, axis=1)
            labels = labels[::num_frames]

        preds = np.argmax(outputs, axis=1)

        assert preds.shape == labels.shape
        for index in range(labels.shape[0]):
            conf_mat[int(labels[index])][preds[index]] += 1

        return conf_mat



    def save_checkpoint(self, state, is_best ,filename='checkpoint.pth.tar',
                        best_filename='model_best.pth.tar'):


        ################# Create Models Dir ###############
        models_dir = os.path.join(self.args.exp_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        ###################################################


        # print (model_dir)
        filename = os.path.join(models_dir, filename)
        best_filename = os.path.join(models_dir, best_filename)
        # -------------------------------------------------
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_filename)



    def load_checkpoint(self, model, optimizer , load_best=False, filename='checkpoint.pth.tar',
                        best_filename='model_best.pth.tar'):

        ################# Create Models Dir ###############
        models_dir = os.path.join(self.args.exp_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        ###################################################

        if not load_best:
            model_file = os.path.join(models_dir, filename)
        else:
            model_file = os.path.join(models_dir, best_filename)

        ############### IF model file Doesnot Exist/Dont Bother loading
        if not os.path.isfile(model_file):
            return

        checkpoint = torch.load(model_file)
        self.best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        self.args.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']



    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * ( self.args.lr_decay ** (epoch // self.args.lr_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr







    #################### Get Mode, Optimizer, Loss-Criteria ##################
    def get_model(self):

        if self.args.architecture in ['resnet34' , 'resnet50' , 'resnet101']:
            if self.args.architecture == 'resnet34':
                model = torchmodels.resnet34(pretrained=True)
            if self.args.architecture == 'resnet50':
                model = torchmodels.resnet50(pretrained=True)
            if self.args.architecture == 'resnet101':
                model = torchmodels.resnet101(pretrained=True)

            num_classes = self.args.num_classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)


        if self.args.architecture in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:

            if self.args.architecture == 'densenet121':
                model = torchmodels.densenet121(pretrained=True)
            if self.args.architecture == 'densenet169':
                model = torchmodels.densenet169(pretrained=True)
            if self.args.architecture == 'densenet201':
                model = torchmodels.densenet201(pretrained=True)
            if self.args.architecture == 'densenet161':
                model = torchmodels.densenet161(pretrained=True)

            num_classes = self.args.num_classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)


        if self.args.architecture in [ 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
        'vgg19_bn', 'vgg19']:
            if self.args.architecture == 'vgg11':
                model = torchmodels.vgg11(pretrained=True)
            if self.args.architecture == 'vgg11_bn':
                model = torchmodels.vgg11_bn(pretrained=True)
            if self.args.architecture == 'vgg13':
                model = torchmodels.vgg13(pretrained=True)
            if self.args.architecture == 'vgg13_bn':
                model = torchmodels.vgg13_bn(pretrained=True)
            if self.args.architecture == 'vgg16':
                model = torchmodels.vgg16(pretrained=True)
            if self.args.architecture == 'vgg16_bn':
                model = torchmodels.vgg16_bn(pretrained=True)
            if self.args.architecture == 'vgg19_bn':
                model = torchmodels.vgg19_bn(pretrained=True)
            if self.args.architecture == 'vgg19':
                model = torchmodels.vgg19(pretrained=True)

            num_classes = self.args.num_classes
            in_features = model.classifier[6].in_features
            n_module = nn.Linear(in_features, num_classes)
            n_classifier = list(model.classifier.children())[:-1]
            n_classifier.append(n_module)
            model.classifier = nn.Sequential(*n_classifier)




        if self.args.cuda:
            model.cuda()

        return model



    def get_optimizer(self):

        if self.args.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay
                                  )
        if self.args.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr = self.args.lr,
                                   betas = (self.args.beta1 , self.args.beta2),
                                   weight_decay=self.args.weight_decay)

        return optimizer


    def get_criterion(self):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def get_tblogger(self):
        from util.logger import Logger
        tb_log_dir = os.path.join(self.args.exp_dir, 'tb_log')
        if os.path.exists(tb_log_dir):
            shutil.rmtree(tb_log_dir)
        if not os.path.exists(tb_log_dir):
            os.makedirs(tb_log_dir)

        print ('tensorboard --logdir=' +  tb_log_dir + ' --port=6006')
        logger = Logger(tb_log_dir)
        return logger







class ClassificationArguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False


    def initialize(self):
        ####### Data Loader Options ##############
        exp_arg = self.parser.add_argument_group('exp', 'experiment specific arguments')

        exp_arg.add_argument("-m", "--mode", metavar='N', default='train',
                             help='mode train/test (default: train)')

        # exp_arg.add_argument("-ss", "--start-set", default=1, type=int, metavar='N',
        #                      help='start set of experiment default(1)')
        # exp_arg.add_argument("-es", "--end-set", default=10, type=int, metavar='N',
        #                      help='end set of experiment default(10)')
        #
        # exp_arg.add_argument("-sr", "--start-run", default=1, type=int, metavar='N',
        #                      help='start set of experiment default(1)')
        # exp_arg.add_argument("-er", "--end-run", default=1, type=int, metavar='N',
        #                      help='end set of experiment default(10)')

        exp_arg.add_argument("-r", "--resume", help="Resuming Experiment after stop",
                             action="store_true")

        exp_arg.add_argument("-cpb", "--copy_back", help="copy back models after every experiment",
                             action="store_true")

        exp_arg.add_argument("-ww", "--which-way", type=int, metavar='N', default=0,
                             help='0:individual samples|1:max of samples|2:sum of samples')

        exp_arg.add_argument("-gi",'--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


        exp_arg.add_argument("-db", '--database', default='mmi', metavar='N',
                             help='database (default=ck)')

        exp_arg.add_argument("-nc", "--num-classes", default=3, type=int, metavar='N',
                             help='number of classes in classification')





        exp_arg.add_argument("-cd", '--cropped_dir', default='cropped_256',
                             metavar='N',
                             help='cropped_dir (default=cropped_256)')

        exp_arg.add_argument("-pd", '--partition_dir', default='six_basic',
                             metavar='N',
                             help='cropped_dir (default=cropped_256)')


        exp_arg.add_argument("-arch", '--architecture', default='vgg16',
                             metavar='A',
                             help='architecture (resnet101|vgg_16|zibonet)')


        exp_arg.add_argument("-lf", '--log_prefix', default='',
                             metavar='N',
                             help='print prefix for logs (default=\'\') (example=\'mmi_set1\')')

        exp_arg.add_argument("-ename", '--experiment_name', default='def',
                             metavar='N',
                             help='experiment name (default=\'\') example=\'emo_baselines\'')

        exp_arg.add_argument("-epad", '--experiment_padding', default='default',
                             metavar='N',
                             help='experiment padding for storing (default=\'\') (example=\'run_1\')')



        ####### Data Loader Options ##############
        dl_arg = self.parser.add_argument_group('dl', 'data loader related arguments')
        # Used in all optimizers
        dl_arg.add_argument( "-nw", '--num-workers',  default=4, type=int,
                             metavar='W', help='num-workers (default: 4)')

        dl_arg.add_argument("-pm", "--pin-memory", help="Pin Memory for CUDA",
                            action="store_true")

        dl_arg.add_argument( "-epl", '--epoch-length',  default=2000, type=int,
                             metavar='W', help='epoch-length (default: 2000)')

        ####### Optimizer Related Options ##############
        opt_arg = self.parser.add_argument_group('opt', 'optimization specific arguments')

        opt_arg.add_argument("-ep", "--epochs", default=35, type=int, metavar='N',
                             help='number of epochs in the experiment default (35)')


        opt_arg.add_argument("-sep", "--start-epoch", default=1, type=int, metavar='N',
                             help='manual epoch number (useful on restarts)')

        opt_arg.add_argument("-ptnc", '--patience', default=0, type=int, metavar='N',
                             help='patience for early stopping'
                                  '(0 means no early stopping)')


        opt_arg.add_argument("-lr", '--lr', default=0.01, type=float,
                             metavar='LR',
                             help='initial learning rate (default: 0.1)')
        opt_arg.add_argument("-lrd", '--lr-decay', default=0.1, type=float, metavar='N',
                             help='decay rate of learning rate (default: 0.4)')
        opt_arg.add_argument("-lrs", "--lr-step", default=10, type=int, metavar='N',
                             help='learning step')


        opt_arg.add_argument("-bs", '--batch-size', default=64, type=int,
                             metavar='N', help='mini-batch size (default: 64)')
        opt_arg.add_argument("-opt", '--optimizer', default='sgd',
                             choices=['sgd', 'rmsprop', 'adam'], metavar='N',
                             help='optimizer (default=sgd)')



        # Parameters for SGD
        opt_arg.add_argument('--momentum', default=0.9, type=float, metavar='M',
                             help='momentum (default=0.9)')
        opt_arg.add_argument('--no_nesterov', dest='nesterov',
                             action='store_false',
                             help='do not use Nesterov momentum')

        # Parameters for rmsprop
        opt_arg.add_argument('--alpha', default=0.99, type=float, metavar='M',
                             help='alpha for ')

        # Parameters for adam
        opt_arg.add_argument('--beta1', default=0.9, type=float, metavar='M',
                             help='beta1 for Adam (default: 0.9)')
        opt_arg.add_argument('--beta2', default=0.999, type=float, metavar='M',
                             help='beta2 for Adam (default: 0.999)')

        # Used in all optimizers
        opt_arg.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                             metavar='W', help='weight decay (default: 1e-4)')





        ####### Log Related Options ##############
        log_arg = self.parser.add_argument_group('log', 'log related arguments')
        # Used in all optimizers
        opt_arg.add_argument( "-li", '--log-interval',  default=100, type=int,
                             metavar='L', help='log-interval (default: 100)')


        ####### Image Path Related
        imp_arg = self.parser.add_argument_group('image_path', 'image_path related arguments')

        imp_arg.add_argument("-imr", '--image_root', default='def',
                             metavar='N',
                             help='root directory of the image')
        imp_arg.add_argument("-imtr", '--train_list', default='def',
                             metavar='N',
                             help='image list training')
        imp_arg.add_argument("-imvl", '--valid_list', default='def',
                             metavar='N',
                             help='image list validation')
        imp_arg.add_argument("-imts", '--test_list', default='def',
                             metavar='N',
                             help='image list test')


        ####### Tensorboard Related Arguments
        tb_arg = self.parser.add_argument_group('tb_logg', 'arguments related to loggin in tensorboard')
        tb_arg.add_argument("-tb", "--tb_log", help="Use Tensorboard For Logging",
                             action="store_true")

        ####### Other Options ##############
        # other_arg = self.parser.add_argument_group('other', 'log related arguments')
        # # Used in all optimizers
        # opt_arg.add_argument( "-etp", '--experiment-type',  metavar='N', default='def',
        #                      help='for faces (faces)')


    def parse(self):
        if not self.initialized:
            self.initialize()

        self.args = self.parser.parse_args()

        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)

        # set gpu ids
        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])

        if len(self.args.gpu_ids) > 0 and \
            torch.cuda.is_available():

            self.args.cuda = True;
        else:
            self.args.cuda = False;

        return self.args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count



class Renset_Pr(nn.Module):
    def __init__(self, original_model , num_classes = 3):
        super(Renset_Pr, self).__init__()

        childrens = list(original_model.children())
        self.feat = nn.Sequential(*childrens[:-2])
        feat_size = self.get_feat_size()
        self.fc_feat      = nn.Linear( feat_size , 512 )
        self.fc_feat_bn   = nn.BatchNorm1d( 512 )
        self.fc = nn.Linear(512 , num_classes)


    def forward(self, x ):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        feat  = self.fc_feat( x)
        feat_bn = self.fc_feat_bn(feat)
        out   = self.fc(feat_bn)

        return out

    def get_feat_size(self ):
        dummy_input = Variable(torch.randn(1, 3, 224, 224))
        feat = self.feat(dummy_input)
        shape = feat.shape
        return int( shape[1]*shape[2]*shape[3])

if __name__ == '__main__':
    parser = ClassificationArguments()
    args = parser.parse()

    if torch.cuda.is_available():
        print ("Cuda Available:")

    main(args)