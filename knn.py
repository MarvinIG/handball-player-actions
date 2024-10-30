# This script was created with the help of ChatGPT
from modelstructure import model_configs

import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from PIL import Image

def start_knn(data="path_to_records", output_csv='knn_benchmark_results.csv', n_neighbors=1):
    """
    Runs KNN analysis on the given data and saves the accuracy results to a CSV file.
    
    Parameters:
    - data: Path to the directory containing the image data.
    - classes: Path to the file containing class labels.
    - output_csv: Filename for the output CSV.
    - n_neighbors: Number of neighbors for the KNN algorithm.
    """

    # Prepare data and labels
    images = []
    labels = []

    # Load images and extract labels from filenames
    for filename in os.listdir(data):
        if filename.endswith('.png'):
            label = filename.split('_')[0]  
            labels.append(label)
            img_path = os.path.join(data, filename)
            img = Image.open(img_path).convert('L').resize((32,32))
            images.append(np.array(img).flatten())  

    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    print('Labels',  set(y))

    # Encode labels to numerical values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2025, stratify=y_encoded)

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {overall_accuracy:.2f}")

    # Calculate per-class accuracy
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, labels=le.transform(le.classes_), output_dict=True
    )
    if not os.path.exists('KNNanalysis'):
        os.mkdir('KNNanalysis')
    # Write the results to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Accuracy', 'Support'])
        
        for class_name, metrics in report.items():
            if class_name != 'accuracy':  # Skip the overall accuracy row
                writer.writerow([class_name, metrics['precision'], metrics['support']])
        
        # Add overall accuracy to the CSV
        writer.writerow(['Overall', overall_accuracy, 'N/A'])

    print(f"Results saved to {output_csv}")



