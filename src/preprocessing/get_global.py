import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import sys

# hardcoded
desired_width = 256

project_dir = os.getcwd()
sys.path.insert(0, project_dir)
src_dir = os.path.join( project_dir , 'data/OneDrive-2018-03-20')
dst_dir = os.path.join( project_dir , 'data/cropped_images/global_256')





# https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
def resize_image(src_dst):
    src_image_path = src_dst[0];
    dst_image_path = src_dst[1];

    # if need to read grayscale need
    img = cv2.imread(src_image_path)
    img_output = cv2.resize(img, (desired_width, desired_width), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst_image_path, img_output)


for parent, dirnames, filenames in os.walk(src_dir):
    for filename in filenames:
        extension = os.path.splitext(filename)[1].strip().lower()
        # print (extension)
        if extension in ['.jpg', '.png']:
            from_file_path = os.path.join(parent, filename)
            dst_file_path = from_file_path.replace(src_dir, dst_dir)
            dst_file_path_dirname = os.path.dirname(dst_file_path)
            if not os.path.exists(dst_file_path_dirname):
                os.makedirs(dst_file_path_dirname)

            # resize_image((from_file_path,dst_file_path))
            resize_image((from_file_path, dst_file_path))



