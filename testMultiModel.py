import os
from multimodel import multiModel
from PIL import Image
from modelstructure import model_configs
def get_relevant_labels_from_filename(img_name, model_labels):
    image_labels = img_name.split('_')
    image_labels.pop()
    if '' in image_labels:
        image_labels.remove('') 
    return list(set(image_labels).intersection(set(model_labels)))

wrongCounter = 0
counter = 0
wronglog = dict()
for model in ['Positions', 'shotType', 'trickShots', 'feints', 'blocks']:
    wronglog[model] = 0

for img_name in os.listdir('all'):
    counter += 1
    #img_name = "RL-Fernwurf_über-Block_Block-Links_Block-Rechts_1729640671437.png"
    full_path = f'all/{img_name}'
    image = Image.open(full_path).convert('RGB')
    result :dict[str, str] = multiModel(image)
    for modelname, modelresult in result.items():
        #print('checking model', modelname, modelresult)
        relevant_file_labels = get_relevant_labels_from_filename(img_name, model_configs[modelname])
        if relevant_file_labels == []:
                relevant_file_labels = ['NONE']
        #print('relevant file labels',relevant_file_labels)
        if modelname == 'blocks':
            modelresult = modelresult.split('_and_')
            #print("blocks model, now having modelresult as", modelresult)
            modelIsCorrect = set(modelresult) == set(relevant_file_labels)
            #print(modelIsCorrect)
            if not modelIsCorrect:
                 wrongCounter += 1
                 wronglog[modelname] +=1
                 break
        else:
            #print('modelresults', modelresult)
            modelIsCorrect = [modelresult] == relevant_file_labels
            #print(modelIsCorrect)
            if not modelIsCorrect:
                 wrongCounter += 1
                 wronglog[modelname] +=1
                 break
    

print('Wrong Evaluations', wrongCounter)
print('Total files tested', counter)
acc =(counter - wrongCounter)/counter
print('Model Accuracy:', acc)





    


    
        
    