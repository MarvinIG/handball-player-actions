import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from collections import Counter
# Function to set up the folder for the analysis
def setup_analysis(modelname):
    directory = f'CNNanalysis/{modelname}/'
    if not os.path.exists('CNNanalysis'):
       print(f"Creating Folder CNNanalysis")
       os.mkdir('CNNanalysis')
    if not os.path.exists(directory):
        print(f"Creating Analysis Folder {directory}")
        os.mkdir(directory)
    else:
        print(f"Removing content from {directory}")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

def earlyExit(path):
    toRet = True
    fileExist = os.path.exists(path)
    if fileExist:
        keepGoing = None
        keepGoing = input(f'Es existiert bereits ein Modell bei\n {path}\n. Trotzdem fortfahren? (y/n)')
        if keepGoing == 'y':
            print("Es wird fortgefahren...")
            return False
        else:
            return True
    else:
        print('No model found, starting training')
    return False

# main routine for training
def start_training(data_dir, MODEL, TRAIN_TRANSFORMS, VAL_TRANSFORMS, LEARNING_RATE, EPOCHS,
                   BATCHSIZE_TRAIN, BATCHSIZE_VAL, MODELNAME, penalizeUnderrep = False):
    """
    By running this function, the training starts immidiately.
    A folder to save the model and to log the accuracy is automatically created at './CNNanalysis/MODELNAME'
    """
    modelname = MODELNAME
    print(f"""Starting training of CNN. Based of the directory, {modelname} is trained.
Properties:
    Learning Rate: {LEARNING_RATE}
    EPOCHS: {EPOCHS}
    BATCHSIZE_TRAIN: {BATCHSIZE_TRAIN}
    BATCHSIZE_VAL: {BATCHSIZE_VAL}
""")
    fullStop = earlyExit(f'CNNanalysis/{modelname}/cnn_model_{modelname}.pth')
    if fullStop is True:
        print("Abbruch!")
        return False
    
    setup_analysis(modelname)

    torch.manual_seed(2001)
    # Directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Transformations
    train_transforms = TRAIN_TRANSFORMS
    val_transforms = VAL_TRANSFORMS

    # Load Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False)

    # Init Model
    num_classes = len(train_dataset.classes)  # Number of classes based on folder structure
    model = MODEL(num_classes)

    # LossFunc and Optimizer
    criterion = nn.CrossEntropyLoss()
    if penalizeUnderrep:
        print("Class names and their order:", train_dataset.classes)  
        print("Weights for CorssEntropy are calculated...")
        
        
        class_counts = Counter([label for _, label in train_dataset.samples])
        total_samples = sum(class_counts.values())
        
       
        class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
        
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights_tensor = torch.tensor(class_weights).to(device)
        
        
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    class_names = train_dataset.classes

    # Training the model
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, log_file=None):
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            # Print and log training information
            train_info = f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
            print(train_info)
            log_file.write(train_info + "\n")
            

            # Validation loop
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = running_val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            val_info = f"Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            print(val_info)
            log_file.write(val_info + "\n")

            # Save class-wise accuracies at specific epochs
            if epoch == 1 or epoch % 5 == 0:
                class_accuracies_train = calculate_class_accuracies(model, train_loader, class_names)
                class_accuracies_val = calculate_class_accuracies(model, val_loader, class_names)
                save_classwise_accuracy_to_csv(epoch, class_accuracies_train, class_accuracies_val, class_names)

            if train_accuracy == 100:
                print("Stopped training")
                
        # Plot the training and validation loss/accuracy curves
        plot_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs)

        print("Training complete.")
        return True

    # Function to calculate class-wise accuracies
    def calculate_class_accuracies(model, loader, classes):
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        class_accuracies = [100 * (correct / total if total > 0 else 0) for correct, total in zip(class_correct, class_total)]
        return class_accuracies

    # Save class-wise accuracy to CSV
    def save_classwise_accuracy_to_csv(epoch, class_accuracies_train, class_accuracies_val, classes, file_name=f'CNNanalysis/{modelname}/class_accuracies_{modelname}.csv'):
        """Save class-wise accuracy at specific epochs to a CSV file with new columns for each epoch."""
        
        df = pd.DataFrame({
            'Class': classes,
            f'Train Accuracy (Epoch {epoch})': class_accuracies_train,
            f'Validation Accuracy (Epoch {epoch})': class_accuracies_val
        })
        
        if os.path.exists(file_name):
            existing_df = pd.read_csv(file_name)
            merged_df = existing_df.merge(df, on='Class', how='outer')
        else:
            merged_df = df

        merged_df.to_csv(file_name, index=False)
        print(f'Class-wise accuracies saved/updated in {file_name}')

    # Plotting functions for losses and accuracies
    def plot_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies, epochs, modelname=modelname):
        """Plot train and validation loss/accuracy graphs and save them as images."""
        epoch_range = np.arange(1, epochs + 1)
        save_dir = f'CNNanalysis/{modelname}'
        os.makedirs(save_dir, exist_ok=True)

        # Plot Losses
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_range, train_losses, label='Train Loss')
        plt.plot(epoch_range, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(save_dir, f'train_val_loss_{modelname}.png')
        plt.savefig(loss_plot_path)
        print(f'Saved train vs validation loss plot to {loss_plot_path}')
        plt.close()

        # Plot Accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_range, train_accuracies, label='Train Accuracy')
        plt.plot(epoch_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train vs Validation Accuracy')
        plt.legend()
        accuracy_plot_path = os.path.join(save_dir, f'train_val_accuracy_{modelname}.png')
        plt.savefig(accuracy_plot_path)
        print(f'Saved train vs validation accuracy plot to {accuracy_plot_path}')
        plt.close()
    with open(f"{MODELNAME}_training_log.txt", "w") as log_file:
        log_file.write(f"Training log for model: {MODELNAME}\n")
        log_file.write("Epoch\tTrain Loss\tTrain Accuracy\tVal Loss\tVal Accuracy\n")

# Then, open the file in append mode to log each epoch
    with open(f"{MODELNAME}_training_log.txt", "a") as log_file:
        # Start training
        train_model(model, train_loader, val_loader, criterion, optimizer, log_file=log_file)

    # Save the trained model
    torch.save(model.state_dict(), f'CNNanalysis/{modelname}/cnn_model_{modelname}.pth')


# Main routine for binary classification training
def start_binary_training(data_dir, MODEL, TRAIN_TRANSFORMS, VAL_TRANSFORMS, LEARNING_RATE, EPOCHS,
                          BATCHSIZE_TRAIN, BATCHSIZE_VAL, MODELNAME, penalizeUnderrep=False):
    """
    Starts the training process for binary classification.
    """
    modelname = MODELNAME
    print(f"""Starting training of CNN for binary classification: {modelname}.
Properties:
    Learning Rate: {LEARNING_RATE}
    EPOCHS: {EPOCHS}
    BATCHSIZE_TRAIN: {BATCHSIZE_TRAIN}
    BATCHSIZE_VAL: {BATCHSIZE_VAL}
""")
    
    setup_analysis(modelname)

    # Directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Transformations
    train_transforms = TRAIN_TRANSFORMS
    val_transforms = VAL_TRANSFORMS

    # Load Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False)

    # Initialize Model
    model = MODEL().to('cuda' if torch.cuda.is_available() else 'cpu')  

    # Loss Function
    if penalizeUnderrep:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.tensor([0.1, 0.9]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])  # Adjust for imbalanced classes
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)

    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training the model
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.float().to('cuda' if torch.cuda.is_available() else 'cpu')  # Convert labels to float for BCEWithLogitsLoss

            optimizer.zero_grad()
            outputs = model(inputs).view(-1)  # Squeeze the output to match labels
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            preds = torch.round(torch.sigmoid(outputs))  # Get binary predictions
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f'Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.float().to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(inputs).view(-1)  # Squeeze the output to match labels
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                preds = torch.round(torch.sigmoid(outputs))  # Get binary predictions
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch}/{EPOCHS}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Save intermediate results
        save_training_progress(epoch, train_losses, val_losses, train_accuracies, val_accuracies, modelname)

    # Save the final model
    torch.save(model.state_dict(), f'CNNanalysis/{modelname}/cnn_model_{modelname}.pth')
    print(f"Model saved to CNNanalysis/{modelname}/cnn_model_{modelname}.pth")

def save_training_progress(epoch, train_losses, val_losses, train_accuracies, val_accuracies, modelname):
    """Save training progress including plots and CSV."""
    epoch_range = np.arange(1, len(train_losses) + 1)
    save_dir = f'CNNanalysis/{modelname}'
    os.makedirs(save_dir, exist_ok=True)

    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_losses, label='Train Loss')
    plt.plot(epoch_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'train_val_loss_{modelname}.png'))
    plt.close()

    # Plot Accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_accuracies, label='Train Accuracy')
    plt.plot(epoch_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'train_val_accuracy_{modelname}.png'))
    plt.close()

    # Save results to CSV
    df = pd.DataFrame({
        'Epoch': epoch_range,
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies
    })
    df.to_csv(os.path.join(save_dir, f'training_progress_{modelname}.csv'), index=False)
    print(f'Training progress saved to CNNanalysis/{modelname}/training_progress_{modelname}.csv')

def start_transfer_learning(data_dir, MODEL, TRAIN_TRANSFORMS, VAL_TRANSFORMS, LEARNING_RATE, EPOCHS, BATCHSIZE_TRAIN, BATCHSIZE_VAL, MODELNAME, class_order):
    """
    Starts training using a transfer learning approach with incremental class inclusion.
    Training starts with the first two classes in 'class_order', then allows the user to include additional classes every 5 epochs.
    """
    modelname = MODELNAME
    print(f"Starting transfer learning of CNN with model {modelname}.")
    print(f"Properties:\n    Learning Rate: {LEARNING_RATE}\n    EPOCHS: {EPOCHS}\n    BATCHSIZE_TRAIN: {BATCHSIZE_TRAIN}\n    BATCHSIZE_VAL: {BATCHSIZE_VAL}\n")

    # Directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Transformations
    train_transforms = TRAIN_TRANSFORMS
    val_transforms = VAL_TRANSFORMS

    # Init Model
    num_classes = len(class_order)  # Total number of classes
    model = MODEL(num_classes)

    # LossFunc and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Progressive class training
    current_classes = class_order[:4]  # Start with the first two classes
    class_inclusion_idx = 4  # Track which class to include next

    # Function to filter datasets based on current classes
    def filter_dataset(dataset, current_classes):
        filtered_indices = [i for i, (_, label) in enumerate(dataset.samples) if dataset.classes[label] in current_classes]
        return torch.utils.data.Subset(dataset, filtered_indices)

    # Load datasets and filter for initial classes
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    train_dataset = filter_dataset(train_dataset, current_classes)
    val_dataset = filter_dataset(val_dataset, current_classes)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False)

    # Training the model
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS):
        nonlocal class_inclusion_idx  # Allows modifying the outer scope variable
        nonlocal current_classes
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

            # Validation loop
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_loss = running_val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            # Ask for new class inclusion every 5 epochs
            if epoch % 5 == 0 and class_inclusion_idx < len(class_order):
                user_input = input(f"Include class '{class_order[class_inclusion_idx]}' into training? (y/n): ")
                if user_input.lower() == 'y':
                    current_classes.append(class_order[class_inclusion_idx])
                    class_inclusion_idx += 1
                    print(f"Now training with classes: {current_classes}")

                    # Update datasets and loaders
                    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
                    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
                    train_dataset = filter_dataset(train_dataset, current_classes)
                    val_dataset = filter_dataset(val_dataset, current_classes)
                    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE_TRAIN, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE_VAL, shuffle=False)

            # Save class-wise accuracies at specific epochs
            #if epoch % 5 == 0:
                #class_accuracies_train = calculate_class_accuracies(model, train_loader, current_classes)
                #class_accuracies_val = calculate_class_accuracies(model, val_loader, current_classes)
                #save_classwise_accuracy_to_csv(epoch, class_accuracies_train, class_accuracies_val, current_classes)

        # Plot the training and validation loss/accuracy curves
        plot_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs)

        print("Training complete.")


    def calculate_class_accuracies(model, loader, classes):
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)
        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        class_accuracies = [100 * (correct / total if total > 0 else 0) for correct, total in zip(class_correct, class_total)]
        return class_accuracies

    def save_classwise_accuracy_to_csv(epoch, class_accuracies_train, class_accuracies_val, classes, file_name=f'CNNanalysis/{modelname}/class_accuracies_{modelname}.csv'):
        """Save class-wise accuracy at specific epochs to a CSV file with new columns for each epoch."""
        
        df = pd.DataFrame({
            'Class': classes,
            f'Train Accuracy (Epoch {epoch})': class_accuracies_train,
            f'Validation Accuracy (Epoch {epoch})': class_accuracies_val
        })
        
        if os.path.exists(file_name):
            existing_df = pd.read_csv(file_name)
            merged_df = existing_df.merge(df, on='Class', how='outer')
        else:
            merged_df = df

        merged_df.to_csv(file_name, index=False)
        print(f'Class-wise accuracies saved/updated in {file_name}')

    def plot_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies, epochs, modelname=modelname):
        """Plot train and validation loss/accuracy graphs and save them as images."""
        epoch_range = np.arange(1, epochs + 1)
        save_dir = f'CNNanalysis/{modelname}'
        os.makedirs(save_dir, exist_ok=True)

        # Plot Losses
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_range, train_losses, label='Train Loss')
        plt.plot(epoch_range, val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(save_dir, f'train_val_loss_{modelname}.png')
        plt.savefig(loss_plot_path)
        print(f'Saved train vs validation loss plot to {loss_plot_path}')
        plt.close()

        # Plot Accuracies
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_range, train_accuracies, label='Train Accuracy')
        plt.plot(epoch_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train vs Validation Accuracy')
        plt.legend()
        accuracy_plot_path = os.path.join(save_dir, f'train_val_accuracy_{modelname}.png')
        plt.savefig(accuracy_plot_path)
        print(f'Saved train vs validation accuracy plot to {accuracy_plot_path}')
        plt.close()

    # Start training
    train_model(model, train_loader, val_loader, criterion, optimizer)

    # Save the trained model
    torch.save(model.state_dict(), f'CNNanalysis/{modelname}/cnn_model_{modelname}.pth')

    print("Model saved and training finished.")

