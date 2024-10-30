import os
import csv
from modelstructure import model_configs, prepare_data_for_model
    
def findMajorityClass(directory_path, output_csv='majority_class.csv'):
    majorClass = {'name': None, 'length': -1}
    total_files = len(os.listdir('./all'))
    print('Total records:', total_files)

    for Class in os.listdir(directory_path):
        class_path = os.path.join(directory_path, Class)
        if os.path.isdir(class_path): 
            classLength = len(os.listdir(class_path))
            if classLength > majorClass['length']:
                majorClass['name'] = Class
                majorClass['length'] = classLength

    percentage = majorClass['length'] / total_files

   
    print('Majority Class:', majorClass['name'])
    print('Length:', majorClass['length'])
    print('Percentage:', percentage)

    # Create CSV
    if not os.path.exists('MCCanalysis'):
        os.mkdir('MCCanalysis')
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Length', 'Percentage'])
        writer.writerow([majorClass['name'], majorClass['length'], percentage])

    print(f'Results saved to {output_csv}')
    



