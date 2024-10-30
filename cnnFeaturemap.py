import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
def load_model(path_to_model, modelclass, classnum):
    # Replace 'path_to_model.pth' with the path to your saved model
    model = modelclass(classnum)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()  # Set the model to evaluation mode
    print(model)
    return model

# Function to register a hook to capture feature maps
def get_feature_maps(name, feature_maps_dict):
    def hook(model, input, output):
        feature_maps_dict[name] = output.detach()
    return hook

# Function to visualize feature maps
def visualize_feature_maps(feature_maps, num_maps=3, model='MODEL', layer='pool'):
    # Get the feature maps from the dictionary
    maps = feature_maps.get(layer)
    
    if maps is None:
        print("No feature maps found for the specified layer.")
        return
    
    # Assuming feature maps are in the shape [batch_size, num_features, height, width]
    num_feature_maps = maps.shape[1]
    num_maps = min(num_maps, num_feature_maps)  # Ensure we don't plot more maps than available

    # Create subplots for visualization
    fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
    for i in range(num_maps):
        axes[i].imshow(maps[0, i].cpu().numpy(), cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Feature Map {i+1}')
    
    plt.savefig(f'CNNAnalysis/{model}/featuremaps_{model}.png')
    

def create_featuremap(path_to_model, path_to_image, modelname, modelclass, modeltransform, layer, classnum, num_maps=3):
    # Load the model
    model = load_model(path_to_model=path_to_model, modelclass=modelclass, classnum=classnum)

    # Dictionary to store feature maps
    feature_maps = {}

    # Register the hook to a specific layer
    # Replace 'layer_name' with the layer from which you want to extract feature maps
    model.pool.register_forward_hook(get_feature_maps(layer, feature_maps))

    # Preprocess the input image
    transform = modeltransform
    image = Image.open(path_to_image)
    # Ensure the image is converted to RGB (3 channels)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    
    with torch.no_grad():
        _ = model(input_image)

    
    visualize_feature_maps(feature_maps, num_maps=num_maps, model=modelname, layer=layer)
