# Datacollection Environment for Handball CNN Training

This repository contains files and scripts for training multiple CNN models on handball action classification for a bachelor's thesis. The environment replicates the training process detailed in the thesis, with `trainingWalkthrough.py` as the main control file. Each function, folder, and output is structured to streamline training and benchmarking while reducing redundancy.

## File Structure and Key Scripts

### `trainingWalkthrough.py`
The main script for training, calling functions for dataset preparation, model training, and benchmark evaluation. Outlined functions streamline CNN training across various configurations, minimizing the need for multiple scripts. Comments on unused code reflect initial model development iterations.

### Dataset Preparation
The `prepare_data_for_model()` function creates organized datasets for each model based on the specified folder structure, typically in `CNN/`. If a structure already exists, the function prompts to confirm re-creation, deleting any existing data in the directory.

### Training Process
The `start_training()` function initiates training, logging accuracy, loss, and class-wise accuracy in `CNNanalysis/modelname`. Saved data includes plots, logs, and model state dictionaries. Training is configured to save the final epoch state.

### Feature Maps
Generated post-training when needed for model analysis. Saved feature maps are stored in `CNNanalysis/modelname`.

### Benchmarks
Benchmarking results are created through functions in `trainingWalkthrough.py` and stored in `BENCHMARKNAMEanalysis`.

## Using the Multi-Model

The `multimodel.py` file integrates the trained models for combined classification, leveraging object-oriented programming. Configure the file path to evaluate performance on specific images across all models simultaneously.
