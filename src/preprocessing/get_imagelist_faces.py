import os
import sys
import numpy as np

def get_area_bb(  bb ):
    x = int(bb[0])
    y = int(bb[1])
    h = int((bb[3] - bb[1]))
    w = int((bb[2] - bb[0]))
    area = h * w
    return area

def main(root, partition_dir, landmarks_dir, confidence_threshold = .90, area_low_threshold = 12, area_high_threshold = 48 ):

    # create the files.
    ftrain = open(os.path.join(partition_dir , 'train.txt'), 'w+')
    fval   = open(os.path.join(partition_dir , 'val.txt')  , 'w+')
    ftest = open(os.path.join(partition_dir , 'test.txt'), 'w+')


    for parent, dirnames, filenames in os.walk(root):
        if len(filenames) > 0 and filenames[0].endswith('.png'):

            filenames = filter(lambda image: image[-4:] == '.png', filenames)
            filenames.sort()


            tokens = parent.split('/')
            p_dir = '/'.join(tokens[-3:])

            if 'Test_Images' in parent:
                p_dir = '/'.join(tokens[-2:])

            cur_landmarks_dir = os.path.join( landmarks_dir , p_dir )
            landmark_file = os.path.join(cur_landmarks_dir , 'landmarks.txt')
            bounding_boxes_file = os.path.join(cur_landmarks_dir , 'bbox.txt')

            assert os.path.exists(landmark_file)
            assert os.path.exists(bounding_boxes_file)

            bounding_boxes = np.loadtxt(bounding_boxes_file, ndmin=2)
            landmarks = np.loadtxt(landmark_file, ndmin=2)
            if len(bounding_boxes) == 0 or \
                    len(landmarks) == 0:
                continue

            # print len(filenames) , len(landmarks)
            # assert (len(filenames) == len(landmarks))
            # bounding_boxes, landmarks, filenames = zip(*sorted(zip(bounding_boxes, landmarks , filenames), \
            #                                         key=lambda p: p[0][4], reverse=True))
            # #thresholding confidence
            # thresholded = filter(lambda p:p[0][4] > confidence_threshold,
            #                             zip(bounding_boxes, landmarks,filenames))
            #
            # # thresholding area
            # thresholded = filter(lambda p:get_area_bb(p[0])  > area_low_threshold, thresholded)
            # thresholded = filter(lambda p: get_area_bb(p[0]) > area_high_threshold, thresholded)
            #
            # if len(thresholded) == 0:
            #     continue

            # bounding_boxes , landmarks, filenames = zip(*thresholded)

            for filename in filenames:


                file_id = int(filename[-7:-4])
                area = get_area_bb(bounding_boxes[file_id])
                confidence = bounding_boxes[file_id][4]
                # print filename, file_id
                if area > area_low_threshold and area < area_high_threshold \
                    and confidence > confidence_threshold:

                    if 'Train' in parent:
                        f = ftrain
                    elif 'Validation' in parent:
                        f = fval
                    elif 'Test_Images' in parent:
                        f = ftest



                    if 'Neutral' in os.path.join(parent, filename):
                        f.write(os.path.join(p_dir, filename) + ' 0' + '\n')
                    elif 'Positive' in os.path.join(parent, filename):
                        f.write(os.path.join(p_dir, filename) + ' 1' + '\n')
                    elif 'Negative' in os.path.join(parent, filename):
                        f.write(os.path.join(p_dir, filename) + ' 2' + '\n')
                    else:
                        f.write(os.path.join(p_dir, filename) + ' 0' + '\n')

    ftrain.close()
    ftest.close()
    fval.close()

    train_lines = list()
    val_lines   = list()

    with open(os.path.join(partition_dir , 'train.txt')) as f:
        train_lines = f.readlines()
    with open(os.path.join(partition_dir , 'val.txt')) as f:
        val_lines = f.readlines()


    file_name_train_val = os.path.join( partition_dir , 'train_val.txt')
    with open(file_name_train_val, 'w') as f:
        for line in train_lines+val_lines:
            f.write(line)


project_dir = os.getcwd()

landmarks_dir = os.path.join( project_dir , 'data/landmarks')
root          = os.path.join( project_dir , 'data/cropped_images/aligned_faces')



partition_dir = os.path.join( project_dir , 'data/image_lists/global_faces_90_g12_l48')
file_name_train = os.path.join(partition_dir,'train.txt')
file_name_val   = os.path.join(partition_dir,'val.txt')
file_name_test  = os.path.join(partition_dir,'test.txt')
file_name_test_orig = os.path.join(partition_dir,'test_orig.txt')
if not os.path.exists(partition_dir):
    os.makedirs(partition_dir)
main(root, partition_dir, landmarks_dir, .90, 12*12, 48*48)

partition_dir = os.path.join( project_dir , 'data/image_lists/global_faces_90_g48')
file_name_train = os.path.join(partition_dir,'train.txt')
file_name_val   = os.path.join(partition_dir,'val.txt')
file_name_test  = os.path.join(partition_dir,'test.txt')
file_name_test_orig = os.path.join(partition_dir,'test_orig.txt')
if not os.path.exists(partition_dir):
    os.makedirs(partition_dir)
main(root, partition_dir, landmarks_dir, .90, 48*48, float('inf'))