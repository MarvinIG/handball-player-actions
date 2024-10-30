import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from PIL import Image
import os

from cnnModels import CNN_Positions_V3, CNN_train_transforms_Positions_V3, CNN_shotType_V2, CNN_val_transforms_shotType_V2, CNN_feints_V1, transformerAttributes, CNN_blocks_V1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model:
    def __init__(self, modelclass, classnum, transform, state_dict, class_names, name):
        self.model = modelclass(classnum)
        self.model.load_state_dict(torch.load(state_dict, weights_only=False))
        self.model.eval()
        self.model = self.model.to(device)

        self.transform = transform
        self.class_names = class_names
        self.name = name
    
    def __call__(self, image, raw=False):
        with torch.no_grad():
            image = self.transform(image)
            image = image.unsqueeze(0) 

            output = self.model(image)
            if raw: return output
            probabilities = F.softmax(output, dim=1) 
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class_name = self.class_names[predicted_class_idx]

            return predicted_class_name

class Evaluator:
    def __init__(self, models: list[Model]):
        self.models = models
    
    def __call__(self, image):
        retDict = {}
        for model in self.models:
            retDict[model.name] = model(image)
        return retDict
    

def positions(state_dict='CNNanalysis/Positions_V3/cnn_model_Positions_V3.pth'):
    return Model(CNN_Positions_V3, 19, CNN_train_transforms_Positions_V3, state_dict, ['7m', 'KL', 'KM', 'KR', 'LA-Eckenaussen', 'LA-Seitenaussen', 'RA-Eckenaussen', 'RA-Seitenaussen', 'RL-Durchbruch-abgedrängt', 'RL-Durchbruch-z.-Tor', 'RL-Fernwurf', 'RM-Durchbruch-abgedrängt', 'RM-Durchbruch-z.-Tor', 'RM-Fernwurf', 'RR-Durchbruch-abgedrängt', 'RR-Durchbruch-z.-Tor', 'RR-Fernwurf', 'Tempogegenstoß', 'direkter-Freiwurf'], 'Positions')

def shotType(state_dict='CNNanalysis/shotType_V2X/cnn_model_shotType_V2X.pth'):
    return Model(CNN_shotType_V2, 6, CNN_val_transforms_shotType_V2, state_dict, ['Hüftwurf', 'NONE', 'Schlagwurf-WAGS', 'Schlagwurf-WAS', 'Sprungwurf', 'fallend'], 'shotType')

def trickShots(state_dict='CNNanalysis/trickShots_V2X/cnn_model_trickShots_V2X.pth'):
    return Model(CNN_shotType_V2, 4, CNN_val_transforms_shotType_V2, state_dict, ['Dreher', 'Heber', 'Kempa', 'NONE'], 'trickShots')

def feints(state_dict='CNNanalysis/feints_weighted/cnn_model_feints_weighted.pth'):
    return Model(CNN_feints_V1, 3, transformerAttributes(CNN_feints_V1.MODELSIZE, 2, None, False), state_dict, ['1gg1-WAGS', '1gg1-WAS', 'NONE'], 'feints')

def blocks(state_dict='CNNanalysis/blocks_V2X/cnn_model_blocks_V2X.pth'):
    return Model(CNN_blocks_V1, 8, transformerAttributes(CNN_blocks_V1.MODELSIZE, 1, None, False), state_dict, ['Block-Links', 'Block-Links_and_Block-Rechts', 'Block-Links_and_Block-Rechts_and_über-Block', 'Block-Links_and_über-Block', 'Block-Rechts', 'NONE', 'über-Block', 'über-Block_and_Block-Rechts'], 'blocks')


multiModel = Evaluator((positions(), shotType(), trickShots(), feints(), blocks()))

image = Image.open('./all/RM-Fernwurf_Block-Links_über-Block_1729640859772.png').convert('RGB') 
#image = positions.transform(image)
#image = image.unsqueeze(0)  


#print(multiModel(image))



