# This script was created with the help of ChatGPT

import os
import shutil
import random
from collections import defaultdict
from typing import List


# Define the directory where all PNG files are located


# Define model configurations
model_configs = {
    'Positions': [
        'LA-Eckenaussen', 'LA-Seitenaussen', 'RA-Eckenaussen', 'RA-Seitenaussen',
        'RL-Fernwurf', 'RL-Durchbruch-z.-Tor', 'RL-Durchbruch-abgedrängt',
        'RR-Fernwurf', 'RR-Durchbruch-z.-Tor', 'RR-Durchbruch-abgedrängt',
        'RM-Fernwurf', 'RM-Durchbruch-z.-Tor', 'RM-Durchbruch-abgedrängt',
        'KM', 'KL', 'KR', 'Tempogegenstoß', '7m', 'direkter-Freiwurf'
    ],
    'shotType': [
        'NONE', 'Schlagwurf-WAS', 'Schlagwurf-WAGS', 'Hüftwurf', 'Sprungwurf', 'fallend'
    ],
    # Shot Times Binaray Classification
    'shotType_SchlagwurfWAS':       ['NONE','Schlagwurf-WAS'],
    'shotType_SchlagwurfWAGS':      ['NONE','Schlagwurf-WAGS'],
    'shotType_Hüftwurf':            ['NONE','Hüftwurf'],
    'shotType_Sprungwurf':          ['NONE','Sprungwurf'],
    'shotType_fallend':             ['NONE','fallend'],

    'trickShots': [
        'NONE', 'Heber', 'Dreher', 'Kempa'
    ],
    'feints': [
        'NONE', '1gg1-WAS', '1gg1-WAGS'
    ],
    'blocks': [
        'NONE', 'Block-Links', 'Block-Rechts', 'über-Block', 
        ('Block-Links', 'Block-Rechts', 'über-Block'),
        ('Block-Links', 'Block-Rechts'),
        ('über-Block', 'Block-Rechts'),
        ('Block-Links', 'über-Block')        
    ]
}

def prepare_data_for_model(model_name: str, mode='splitData', saveDir="", source_dir = './all/', noneRatio = 1, weightVal = False):
    """
    The mode parameter can be set to 'reduceNone', which will reduce the None classes in the dataset
    used for the training of the model to be the number of not-None labels.
    The 'allTrain' mode will put every record into the training data (used for the Majority Class Classification)
    weightVal on True will put the remaining None objects into the validation data to better validate the models performance
    Attention: this functions deletes all contents in saveDir when its in the CNN Folder
    """
    random.seed(2001)
    if saveDir == "":
        answer = input("No saveDir given, you want to create the folder structure right where you are? (y/n)")
        if answer != 'y':
            print("Aborted")
            return
    elif saveDir.startswith('CNN') and os.path.exists(saveDir):
        delete_existing_structure = input('Theres already training data found. You want to recreate them? (y/n)')
        if delete_existing_structure == 'y':
            print(f"Removing content from {saveDir}")
            for filename in os.listdir(saveDir):
                file_path = os.path.join(saveDir, filename)
                try:
                    print("Remove", file_path)
                    shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
        else:
            print('Preparation skipped')
            return

    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' is not defined in the configurations.")

    # Get the labels for the specified model
    labels = model_configs[model_name]

    # Create the target directory structure
    target_dir = f'{saveDir}/'
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')

    # Create directories for train and val
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create subdirectories for each label (and combinations if applicable)
    for label in labels:
        label_folder_name = '_and_'.join(label) if isinstance(label, tuple) else label
        os.makedirs(os.path.join(train_dir, label_folder_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label_folder_name), exist_ok=True)

    # Create 'NONE' folder if it exists in the model labels
    if 'NONE' in labels:
        os.makedirs(os.path.join(train_dir, 'NONE'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'NONE'), exist_ok=True)

    # Gather all files and categorize them by labels
    label_to_files = defaultdict(list)
    none_files = []

    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            label_part = filename.rsplit('_', 1)[0]
            file_labels = set(label_part.split('_'))
            
            matched = False

            # Find the appropriate labels for the given model
            
            for label in labels:
                
                if isinstance(label, tuple):
                    # Check for combinations and ensure all labels in the tuple are in the file
                    if set(('Block-Links', 'Block-Rechts', 'über-Block')) in set(file_labels):
                        pass
                    if set(label).issubset(file_labels):
                        label_folder_name = '_and_'.join(label)
                        label_to_files[label_folder_name].append(filename)
                        matched = True
                        break
                else:
                    # Check if the single label is present and not part of a combination
                    if label in file_labels and not any(
                        set(combination).issubset(file_labels) 
                        for combination in labels if isinstance(combination, tuple)
                    ):
                        label_to_files[label].append(filename)
                        matched = True
                        break

            # If no match was found, add the file to 'NONE' if applicable
            if not matched and 'NONE' in labels:
                none_files.append(filename)

    
    total_labeled_files = sum(len(files) for files in label_to_files.values())
    if mode != 'reduceNone':
        total_labeled_files = len(none_files) + total_labeled_files
    split = 0.8
    print("Files in training (not NONE):", total_labeled_files)
    max_none_files = int(noneRatio * total_labeled_files)

    # Randomly select up to 'max_none_files' for the 'NONE' category
    if 'NONE' in labels:
        random.shuffle(none_files)
        label_to_files['NONE'].extend(none_files[:max_none_files])

    
    if mode == 'allTrain': # For MCC
        split = 1
    for label, files in label_to_files.items():
        random.shuffle(files)
        split_idx = int(split * len(files))
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        # Copy files to respective folders
        for file in train_files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, label, file))
        for file in val_files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, label, file))
    
    if weightVal:
        for file in label_to_files['NONE']:
            shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, 'None', file))
    # Print out labels with fewer than 15 images
    print("Labels with fewer than 15 images:")
    for label, files in label_to_files.items():
        if len(files) < 15:
            print(f"{label}: {len(files)} images")

    print(f'Data prepared for model: {model_name}')



# Example usage:
# prepare_data_for_model('Positions')
# prepare_data_for_model('shotType')
# prepare_data_for_model('trickShots')
# prepare_data_for_model('feints')
# prepare_data_for_model('blocks')
