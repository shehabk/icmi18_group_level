#!/usr/bin/env bash

#### 1 get mtcnnn landmarks
python src/preprocessing/get_landmark_mtcnn.py

#### 2 get global images and generate imagelists
python src/preprocessing/get_global.py
python src/preprocessing/get_imagelist_global.py

#### 3 get heatmap images and generate imagelists
python src/preprocessing/get_heatmap_global.py
python src/preprocessing/get_imagelist_heatmap.py

#### 4 get blurred images and generate imagelists
python src/preprocessing/get_blurred_global.py
python src/preprocessing/get_imagelist_blurred.py

#### 5 get aligned faces and generate imagelists
python src/preprocessing/get_aligned_faces.py
python src/preprocessing/get_imagelist_faces.py
