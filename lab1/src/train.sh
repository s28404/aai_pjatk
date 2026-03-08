#!/bin/bash

python main.py --hidden-layers 16 --epochs 100 --version "1_layer_16_100"

python main.py --hidden-layers 16 --epochs 200 --version "1_layer_16_200"

python main.py --hidden-layers 32 16 --epochs 100 --version "2_layers_32_16_100"

python main.py --hidden-layers 32 16 --epochs 200 --version "2_layers_32_16_200"