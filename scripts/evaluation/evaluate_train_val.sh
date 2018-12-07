#!/usr/bin/env bash

# First get the global prediction
python src/evaluation/store_global_prediction.py \
    --trained_on 'train_val' --test_data 'test' \
    --choose_last_model

# Then get the global blurred  prediction
python src/evaluation/store_blurred_prediction.py \
    --trained_on 'train_val' --test_data 'test' \
    --choose_last_model


# Then get the global heatmap prediction
python src/evaluation/store_heatmap_prediction.py \
    --trained_on 'train_val' --test_data 'test' \
    --choose_last_model


# Generate intermediate result for the face stream
python src/evaluation/store_faces_prediction.py \
    --trained_on 'train_val' --test_data 'test' \
    --choose_last_model


# Combine the intermediate faces prediction
python src/evaluation/combine_faces_prediction.py \
    --trained_on 'train_val' --test_data 'test' \
    --choose_last_model


