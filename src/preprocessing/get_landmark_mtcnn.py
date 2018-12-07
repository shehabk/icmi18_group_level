import os
import sys
from PIL import Image
import numpy as np


project_dir = os.getcwd()
sys.path.insert( 0 , project_dir )
src_root    = os.path.join( project_dir , 'data/OneDrive-2018-03-20' )
dst_root    = os.path.join( project_dir , 'data/landmarks')
mtcnn_dir   = os.path.join( project_dir , 'src/mtcnn')


# os.chdir(mtcnn_dir)
# sys.path.append(mtcnn_dir)
open( os.path.join( mtcnn_dir , '__init__.py') , 'w+').close()
from src.mtcnn.src import detect_faces, show_bboxes




def get_landmarks( input_tuple):

    image_path = input_tuple[0]
    landmark_file= input_tuple[1]
    bounding_boxes_file= input_tuple[2]

    print (image_path)
    if os.path.isfile(landmark_file) and \
        os.path.isfile(bounding_boxes_file):
        return

    img = Image.open(image_path)
    img = img.convert('RGB')
    # Already Done processing

    try:
        bounding_boxes, landmarks = detect_faces(img)
    except ValueError:
        return

    np.savetxt(landmark_file , landmarks)
    np.savetxt(bounding_boxes_file, bounding_boxes)
#
#
image_paths = []
landmark_files = []
bounding_boxes_files = []
os.chdir(mtcnn_dir)
for parent, dirnames, filenames in os.walk(src_root):
    if len(filenames) > 0 and filenames[0].endswith('.jpg'):
        filenames = filter(lambda image: image[-4:] == '.jpg', filenames)

        tokens = parent.split('/')
        if tokens[-1] == 'Test_Images':
            p_dir = '/'.join(tokens[-1:])
        else:
            p_dir = '/'.join(tokens[-2:])

        for filename in filenames:
            dst_dir_name = filename[:-4]
            # print dst_dir_name
            image_path = os.path.join(parent , filename )
            dst_dir = os.path.join(dst_root , p_dir , dst_dir_name)

            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            landmark_file = os.path.join(dst_dir , 'landmarks.txt')
            bounding_boxes_file = os.path.join(dst_dir , 'bbox.txt')

            image_paths.append(image_path)
            landmark_files.append(landmark_file)
            bounding_boxes_files.append(bounding_boxes_file)

            # get_landmarks(image_path , landmark_file , bounding_boxes_file)
            # print os.path.dirname(bounding_box_file)
#
for i in range(len(image_paths)):
    get_landmarks((image_paths[i] , landmark_files[i] , bounding_boxes_files[i]))

# pool = ThreadPool(3)
# pool.map(get_landmarks, zip(image_paths, landmark_files , bounding_boxes_files ))
# pool.close();
# pool.join();