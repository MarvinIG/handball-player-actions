import os
import csv
from collections import Counter

def countLabels():
    # Define the directory where your PNG files are located
    directory = './all/'

    # Initialize a Counter to count the occurrences of each label
    label_counter = Counter()

    # Loop through the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # Remove the timestamp part (after the last "_")
            label_part = filename.rsplit('_', 1)[0]
            # Split the labels by "_"
            labels = label_part.split('_')
            # Count each label
            label_counter.update(labels)

    # Save the results to a CSV file
    output_file = 'label_count.csv'
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        # Write the header
        writer.writerow(['label', 'count'])
        # Write the label counts
        for label, count in label_counter.items():
            writer.writerow([label, count])

    print(f'Label counts saved to {output_file}')
