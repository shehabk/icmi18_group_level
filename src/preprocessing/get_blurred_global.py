import warnings
warnings.filterwarnings("ignore")
import os
import sys
import cv2
import numpy as np


project_dir = os.getcwd()
sys.path.insert( 0 , project_dir )
src_root        = os.path.join( project_dir , 'data/OneDrive-2018-03-20' )
landmarks_root  = os.path.join( project_dir , 'data/landmarks')
dst_root        = os.path.join( project_dir , 'data/cropped_images/global_blurred_256')



def blur_faces( input_tuple):
    threshold = .80
    size  = 256
    width = size
    height = size

    image_path = input_tuple[0]
    landmark_file= input_tuple[1]
    bounding_boxes_file= input_tuple[2]
    dst_image_path = input_tuple[3]

    # print image_path

    if not os.path.isfile(landmark_file) or \
        not os.path.isfile(bounding_boxes_file):
        return

    # Already Done
    if os.path.isfile(dst_image_path):
        return


    image = cv2.imread( image_path )
    orig_shape = image.shape

    image = cv2.resize(image , (size , size ) )


##################################
    bounding_boxes = np.loadtxt(bounding_boxes_file, ndmin=2)
    landmarks      = np.loadtxt(landmark_file, ndmin=2)

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
    scalex = ( width  /  float(orig_shape[1]))
    scaley = ( height /  float(orig_shape[0]))

    biggest_height = 0
    for (i, bb) in enumerate(bounding_boxes):
        x = int(bb[0]*scalex)
        y = int(bb[1]*scaley)
        h = int((bb[3] - bb[1])*scaley)
        w = int((bb[2] - bb[0])*scalex)

        if h>biggest_height:
            biggest_height = h

    # Find the kernel size
    kernel_size = biggest_height // 2
    if kernel_size % 2 == 0:
        kernel_size = kernel_size - 1

    image_blur = cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=kernel_size / 3)

    for (i, bb) in enumerate(bounding_boxes):
        x = int(bb[0] * scalex)
        y = int(bb[1] * scaley)
        h = int((bb[3] - bb[1]) * scaley)
        w = int((bb[2] - bb[0]) * scalex)
        image[y:y + h, x:x + w] = image_blur[y:y + h, x:x + w]


    cv2.imwrite(dst_image_path , image)



image_paths = []
dst_image_paths = []
landmark_files = []
bounding_boxes_files = []


for parent, dirnames, filenames in os.walk(src_root):
    if len(filenames) > 0 and filenames[0].endswith('.jpg'):
        filenames = filter(lambda image: image[-4:] == '.jpg', filenames)

        tokens = parent.split('/')
        p_dir = '/'.join(tokens[-2:])

        if 'Test_Images' in parent:
            p_dir = '/'.join(tokens[-1:])


        for filename in filenames:
            image_id = filename[:-4]
            # print dst_dir_name
            image_path = os.path.join(parent , filename )
            lm_dir     = os.path.join(landmarks_root , p_dir , image_id)
            dst_dir    = os.path.join(dst_root , p_dir )
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            dst_image_path = os.path.join( dst_dir , filename )

            landmark_file   = os.path.join(lm_dir , 'landmarks.txt')
            bounding_boxes_file  = os.path.join(lm_dir , 'bbox.txt')

            image_paths.append(image_path)
            dst_image_paths.append(dst_image_path)
            landmark_files.append(landmark_file)
            bounding_boxes_files.append(bounding_boxes_file)


            # get_landmarks(image_path , landmark_file , bounding_boxes_file)
            # print os.path.dirname(bounding_box_file)

for i in range(len(image_paths)):
    blur_faces((image_paths[i] , landmark_files[i] , bounding_boxes_files[i] , dst_image_paths[i]))
    # if i == 10:
    #     break
