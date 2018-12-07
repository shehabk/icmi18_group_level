import os
import sys



def main(root, file_name_train , file_name_val , file_name_test ):
    ftrain = open(file_name_train, 'w+')
    fval   = open(file_name_val  , 'w+')
    ftest  = open(file_name_test , 'w+')
    # ftest_orig = open(file_name_test_orig, 'w+')


    for parent, dirnames, filenames in os.walk(root):
        if len(filenames) > 0 and filenames[0].endswith('.jpg'):

            filenames = filter(lambda image: image[-4:] == '.jpg', filenames)
            filenames.sort()


            tokens = parent.split('/')
            p_dir = '/'.join(tokens[-2:])

            if 'Test_Images' in parent:
                p_dir = '/'.join(tokens[-1:])

            for filename in filenames:

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
    fval.close()
    ftest.close()

    train_lines = list()
    val_lines   = list()

    with open(file_name_train) as f:
        train_lines = f.readlines()
    with open(file_name_val) as f:
        val_lines = f.readlines()

    base_dir = os.path.dirname( file_name_train)
    file_name_train_val = os.path.join( base_dir , 'train_val.txt')
    with open(file_name_train_val, 'w') as f:
        for line in train_lines+val_lines:
            f.write(line)


project_dir = os.getcwd()
sys.path.insert( 0 , project_dir )
partition_dir = os.path.join( project_dir , 'data/image_lists/global_heatmap_256')
root          = os.path.join( project_dir , 'data/cropped_images/global_heatmap_256')


file_name_train = os.path.join(partition_dir,'train.txt')
file_name_val   = os.path.join(partition_dir,'val.txt')
file_name_test  = os.path.join(partition_dir,'test.txt')
# file_name_test_orig = os.path.join(partition_dir,'test_orig.txt')

if not os.path.exists(partition_dir):
    os.makedirs(partition_dir)

main(root, file_name_train , file_name_val , file_name_test)