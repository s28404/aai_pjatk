#!/bin/bash

python main.py --model_type mlp --num_epochs 100 --model_version "v1"
python main.py --model_type mlp --num_epochs 200 --model_version "v1"
python main.py --model_type mlp --num_epochs 100 --model_version "v2"
python main.py --model_type mlp --num_epochs 200 --model_version "v2"

python main.py --model_type knn --n_neighbors 1
python main.py --model_type knn --n_neighbors 3
python main.py --model_type knn --n_neighbors 5
python main.py --model_type knn --n_neighbors 10