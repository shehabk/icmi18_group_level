import warnings
warnings.filterwarnings("ignore")
import os
import sys
import cv2
import numpy as np
from PIL import Image


project_dir = os.getcwd()
sys.path.insert( 0 , project_dir )
src_root        = os.path.join( project_dir , 'data/OneDrive-2018-03-20' )
landmarks_root  = os.path.join( project_dir , 'data/landmarks')
dst_root        = os.path.join( project_dir , 'data/cropped_images/aligned_faces')


from src.util.face_aligner import FaceAlignerMTCNN

def get_faces( input_tuple):

    # threshold           = .80
    # area_threshold      =  48*48

    image_path = input_tuple[0]
    landmark_file= input_tuple[1]
    bounding_boxes_file= input_tuple[2]
    dst_dir = input_tuple[3]

    print image_path
    if not os.path.isfile(landmark_file) or \
        not os.path.isfile(bounding_boxes_file):
        return

    img = Image.open(image_path)
    img = img.convert('RGB')


    ################ Loading and filtering bbox #######################
    bounding_boxes = np.loadtxt(bounding_boxes_file,ndmin=2)
    landmarks      = np.loadtxt(landmark_file , ndmin=2)
    if len(bounding_boxes)==0 or \
            len(landmarks)==0:
        return


    ####################################################################

    #### No thresholding here, will do while creating image list
    ######################################
    # bounding_boxes, landmarks = zip(*sorted(zip(bounding_boxes, landmarks), \
    #                                     key=lambda p: p[0][4], reverse=True)
    # # thresholding confidence
    # thresholded = filter(lambda p:p[0][4] > threshold,
    #                                 zip(bounding_boxes, landmarks))
    # # thresholding area
    # # thresholded = filter(lambda p:get_area_bb(p[0]) > area_threshold, thresholded)
    #
    # if len(thresholded) == 0:
    #     return
    #
    # bounding_boxes , landmarks = zip(*thresholded)
    ######################################


    for (i, bb) in enumerate(bounding_boxes):
        name = "face_%03d.png"%(i)
        full_path = os.path.join(dst_dir , name )
        if os.path.isfile(full_path):
            continue
        face = crop_alligned(img , bb , landmarks[i] )
        if face is not None:
            cv2.imwrite(full_path , face )
        # print full_path

    # bb = bounding_boxes[5]
    # lm = landmarks[5]
    # face1 = crop_alligned(img , bb , lm )
    # face2 = crop_unalligned(img , bb , lm)
    #
    # fig, (ax1 , ax2 ) = plt.subplots(1,2)
    # ax1.imshow(face1)
    # ax2.imshow(face2)
    # plt.show()

    # print bounding_boxes


def get_area_bb(  bb ):
    x = int(bb[0])
    y = int(bb[1])
    h = int((bb[3] - bb[1]))
    w = int((bb[2] - bb[0]))
    area = h * w
    return area

def crop_unalligned( img , bb , lm , new_size = (256,256) ):
    bb = tuple(bb[:4])
    cr = img.crop( bb )
    cr_rz = cr.resize( new_size , Image.BILINEAR)

    cr_rz = np.array(cr_rz)
    cr_rz = cr_rz[:, :, ::-1].copy()

    return cr_rz

def crop_alligned( img , bb , lm,  new_size = (256,256) ):
    img = np.array(img)
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()
    fa = FaceAlignerMTCNN(desiredFaceWidth=256)
    img  = fa.align(img, lm)

    return img

image_paths = []
landmark_files = []
bounding_boxes_files = []
dst_dirs = []

for parent, dirnames, filenames in os.walk(src_root):
    if len(filenames) > 0 and filenames[0].endswith('.jpg'):
        filenames = filter(lambda image: image[-4:] == '.jpg', filenames)

        tokens = parent.split('/')
        p_dir = '/'.join(tokens[-2:])

        if 'Test_Images' in parent:
            p_dir = '/'.join(tokens[-1:])
        # Ignoring Everthing other than test data
        # if not 'Test_Orig' in p_dir:
        #     continue

        for filename in filenames:
            dst_dir_name = filename[:-4]
            # print dst_dir_name
            image_path = os.path.join(parent , filename )
            lm_dir = os.path.join(landmarks_root , p_dir , dst_dir_name)
            dst_dir = os.path.join(dst_root , p_dir , dst_dir_name)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            landmark_file = os.path.join(lm_dir , 'landmarks.txt')
            bounding_boxes_file = os.path.join(lm_dir , 'bbox.txt')

            image_paths.append(image_path)
            landmark_files.append(landmark_file)
            bounding_boxes_files.append(bounding_boxes_file)

            dst_dirs.append(dst_dir)
            # get_landmarks(image_path , landmark_file , bounding_boxes_file)
            # print os.path.dirname(bounding_box_file)

for i in range(len(image_paths)):
    get_faces((image_paths[i] , landmark_files[i] , bounding_boxes_files[i] , dst_dirs[i]))
    # if( i == 5):
    #     break

# pool = ThreadPool(3)
# pool.map(get_landmarks, zip(image_paths, landmark_files , bounding_boxes_files ))
# pool.close();
# pool.join();