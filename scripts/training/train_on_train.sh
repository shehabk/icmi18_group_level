#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1

##### 1 Train Global
python src/training/global_classification.py \
-arch vgg16_bn -ep 32 -bs 45 -lr 0.01 -lrs 8 -nc 3 -ename models/global/train \
-imr  data/cropped_images/global_256 \
-imtr data/image_lists/global_256/train.txt \
-imvl data/image_lists/global_256/val.txt \
-imts data/image_lists/global_256/val.txt -m train


###### 2 Train Heatmap
python src/training/global_classification_heatmap.py \
-arch vgg16_bn -ep 45 -bs 45 -lr 0.01 -lrs 10 -nc 3 -ename models/global_heatmap/train  \
-imr  data/cropped_images/global_256 \
-hr data/cropped_images/global_heatmap_256 \
-imtr data/image_lists/global_heatmap_256/train.txt \
-imvl data/image_lists/global_heatmap_256/val.txt \
-imts data/image_lists/global_heatmap_256/val.txt -m train


###### 3 Train Blurred
python src/training/global_classification_blurred.py \
-arch vgg16_bn -ep 32 -bs 20 -lr 0.01 -lrs 8 -nc 3 -ename models/global_blurred/train \
-imr  data/cropped_images/global_blurred_256 \
-imtr data/image_lists/global_blurred_256/train.txt \
-imvl data/image_lists/global_blurred_256/val.txt \
-imts data/image_lists/global_blurred_256/val.txt -m train


###### 4 Train Faces

start=1
end=10

for epad_number in `seq $start $end`; do

epad=run${epad_number}
#-epad ${epad} \

python src/training/faces_classification.py \
-arch resnet_i24_34 -ep 32 -bs 100 -lr 0.01 -lrs 8 -epl 15000 \
-ename models/faces/train \
-nc 3 \
-epad ${epad} \
-imr data/cropped_images/aligned_faces \
-imtr data/image_lists/global_faces_90_g12_l48/train.txt \
-imvl data/image_lists/global_faces_90_g12_l48/val.txt \
-imts data/image_lists/global_faces_90_g12_l48/val.txt

python src/training/faces_classification.py \
-arch resnet_i48_18 -ep 32 -bs 100 -lr 0.01 -lrs 8 -epl 15000 \
-ename models/faces/train \
-nc 3 \
-epad ${epad} \
-imr data/cropped_images/aligned_faces \
-imtr data/image_lists/global_faces_90_g48/train.txt \
-imvl data/image_lists/global_faces_90_g48/val.txt \
-imts data/image_lists/global_faces_90_g48/val.txt

done