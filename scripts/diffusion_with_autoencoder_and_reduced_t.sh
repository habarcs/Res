#!/bin/sh

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)


config_file=$(dirname "$dir")/configs/diffusion_with_autoencoder_and_reduced_t.py
target=$(dirname "$dir")/config.py
train_file=$(dirname "$dir")/train.py

echo Copying "$config_file" to "$target"
cp "$config_file" "$target"

echo Starting training
python "$train_file"
