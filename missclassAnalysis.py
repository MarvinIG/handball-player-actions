import os
import shutil
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def document_misclassifications(model, data_dir, val_transforms, output_csv='misclassified_files.csv', wrong_label_dir='CNN_wronglabelled'):
    """
    Document misclassified files from the validation set into a CSV file
    and save copies of misclassified images to a specified folder.
    
    Args:
        model (nn.Module): The trained model.
        data_dir (str): Directory containing the validation images.
        val_transforms (transforms.Compose): Transformations to apply to the validation images.
        output_csv (str): Path to save the CSV file documenting misclassifications.
        wrong_label_dir (str): Directory to save copies of misclassified images.
    """
    # Load the validation dataset
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # Batch size of 1 for single image processing
    
    # Get class names from the dataset
    class_names = val_dataset.classes
    
    # Ensure the model is in evaluation mode
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the directory for saving misclassified images if it doesn't exist
    os.makedirs(wrong_label_dir, exist_ok=True)

    # List to store misclassification data
    misclassifications = []

    # Iterate over the validation dataset
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Get the model prediction
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Compare predicted class with the true class
            if predicted.item() != labels.item():
                # Get the file path for the current image from val_dataset.samples
                img_path, true_label = val_dataset.samples[idx]
                
                # Append misclassification details
                misclassifications.append({
                    'file_name': os.path.basename(img_path),
                    'true_class': class_names[true_label],
                    'predicted_class': class_names[predicted.item()]
                })

                # Copy the misclassified image to the 'CNN_wronglabelled' folder
                dst_path = os.path.join(wrong_label_dir, os.path.basename(img_path))
                shutil.copy(img_path, dst_path)
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(misclassifications)

    # Save DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Misclassification details saved to {output_csv}")
    print(f"Misclassified images copied to {wrong_label_dir}")
