import warnings
warnings.filterwarnings("ignore")
import os
import sys
import cv2
import numpy as np
import math


project_dir = os.getcwd()
sys.path.insert( 0 , project_dir )
src_root        = os.path.join( project_dir , 'data/OneDrive-2018-03-20' )
landmarks_root  = os.path.join( project_dir , 'data/landmarks')
dst_root        = os.path.join( project_dir , 'data/cropped_images/global_heatmap_256')



def draw_heatmap( input_tuple):
    threshold = .94
    size  = 256
    width = size
    height = size

    image_path = input_tuple[0]
    landmark_file= input_tuple[1]
    bounding_boxes_file= input_tuple[2]
    dst_image_path = input_tuple[3]

    # print image_path

    # Already Done
    if os.path.isfile(dst_image_path):
        return

    no_faces  = False
    if not os.path.isfile(landmark_file) or \
        not os.path.isfile(bounding_boxes_file):
        no_faces  = True
    else:
        bounding_boxes = np.loadtxt(bounding_boxes_file, ndmin=2)
        landmarks      = np.loadtxt(landmark_file, ndmin=2)
        # print image_path
        # print (len(landmarks))
        if len(landmarks) == 0:
            no_faces = True
        else:
            bounding_boxes , landmarks = zip(*sorted(zip(bounding_boxes,landmarks ),\
                                                     key=lambda p:p[0][4] , reverse=True))
            thresholded = filter(lambda p:p[0][4] > threshold,
                                            zip(bounding_boxes, landmarks))
            if len(thresholded) == 0:
                no_faces = True
            else:
                bounding_boxes , landmarks = zip(*thresholded)







    image = cv2.imread( image_path ,  0 )
    orig_shape = image.shape


    image_gauss_empty  = np.zeros(( height , width) , dtype=np.float32)
    image_gauss        = np.zeros(( height , width) , dtype=np.float32)
##################################

    if no_faces==False:

        scalex = ( width  /  float(orig_shape[1]))
        scaley = ( height /  float(orig_shape[0]))

        for bb in bounding_boxes:
            bb[0] = bb[0]*scalex
            bb[2] = bb[2]*scalex

            bb[1] = bb[1]*scaley
            bb[3] = bb[3]*scaley

            cv2.rectangle(image, (int(bb[0]), int(bb[1])),
                          (int(bb[2]), int(bb[3])), (0, 255, 0), 2)

            ul = ( bb[0] , bb[1])
            br = ( bb[2] , bb[3])
            center = tuple( map( lambda x,y: x + ( y - x )/2.0 , ul , br ))


            sigma   =  int((center[1] -  ul[1])/2.2)
            # sigmay  =  int((center[1] -  ul[1])/2.5)
            # sigmax  =  int((center[0] -  ul[0])/2.5)

            temp_gauss = draw_gaussian(image_gauss_empty.copy(),center, \
                                              sigma = sigma )

            # temp_gauss = fan.draw_gaussian_xy(image_gauss_empty.copy(),center, \
            #                                   sigmax=sigmax , sigmay=sigmay)
            image_gauss = np.maximum(temp_gauss , image_gauss)


    cv2.imwrite(dst_image_path , np.uint8(image_gauss*255))


def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] >
        image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size )
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) -
           int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) -
           int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
    ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    image[image > 1] = 1
    return image




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



print ( len(image_paths))
for i in range(len(image_paths)):
    draw_heatmap((image_paths[i] , landmark_files[i] , bounding_boxes_files[i] , dst_image_paths[i]))
    # if i == 5:
    #     break
