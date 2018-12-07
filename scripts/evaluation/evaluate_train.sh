#!/usr/bin/env bash

# First get the global prediction
python src/evaluation/store_global_prediction.py

# Then get the global blurred  prediction
python src/evaluation/store_blurred_prediction.py

# Then get the global heatmap prediction
python src/evaluation/store_heatmap_prediction.py

# Generate intermediate result for the face stream
python src/evaluation/store_faces_prediction.py

# Combine the intermediate faces prediction.
python src/evaluation/combine_faces_prediction.py

#Combine the intermediate faces prediction.
python src/evaluation/get_final_prediction.py

# Prepare submision same way as the organizer.
# python src/evaluation/prepare_submission.py