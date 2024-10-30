import cv2
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
random.seed(2025)
def extract_red_region(image, output_size=(320, 320)):
    """
    Extracts the red region from an image, finds its bounding box, and scales it
    to fill the entire output size.
    
    Args:
        image (PIL.Image): The input image in PIL format.
        output_size (tuple): The size of the output image (width, height).
        
    Returns:
        PIL.Image: The processed image with only the red region scaled to the output size.
    """
    # Ensure image is in RGB format (remove alpha channel if present)
    image = image.convert("RGB")
    image = np.array(image)
    original_height, original_width = image.shape[:2]

    # Convert the image to the HSV color space for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the range for red color in HSV (adjust these values if needed)
    lower_red = np.array([0, 100, 100])  # Lower bound for red color
    upper_red = np.array([10, 255, 255])  # Upper bound for red color

    # Create a mask to isolate red regions
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Find contours of the red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the bounding box of all the red regions combined
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        
        # Crop the region inside the bounding box
        cropped_red_region = image[y:y+h, x:x+w]

        # Resize the cropped region to the output size
        resized_red_region = cv2.resize(cropped_red_region, output_size, interpolation=cv2.INTER_LINEAR)

        # Convert back to PIL format
        result_image = Image.fromarray(resized_red_region)
    else:
        # If no red regions are found, return a blank image
        result_image = Image.new('RGB', output_size, (255, 255, 255))

    return result_image

def extract_red_region_and_remove_black(image, output_size=(320, 320)):
    """
    Extracts the red region from an image, finds its bounding box, sets non-red pixels to white,
    and scales it to fill the entire output size.
    
    Args:
        image (PIL.Image): The input image in PIL format.
        output_size (tuple): The size of the output image (width, height).
        
    Returns:
        PIL.Image: The processed image with only the red region visible, scaled to the output size.
    """
    # Ensure image is in RGB format (remove alpha channel if present)
    image = image.convert("RGB")
    image = np.array(image)
    original_height, original_width = image.shape[:2]

    # Convert the image to the HSV color space for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the range for red color in HSV (adjust these values if needed)
    lower_red = np.array([0, 100, 100])  # Lower bound for red color
    upper_red = np.array([10, 255, 255])  # Upper bound for red color

    # Create a mask to isolate red regions
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Create an output image where non-red regions are white
    red_only_image = image.copy()
    red_only_image[mask == 0] = [255, 255, 255]  # Set non-red pixels to white

    # Find contours of the red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the bounding box of all the red regions combined
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        
        # Crop the region inside the bounding box
        cropped_red_region = red_only_image[y:y+h, x:x+w]

        # Resize the cropped region to the output size
        resized_red_region = cv2.resize(cropped_red_region, output_size, interpolation=cv2.INTER_LINEAR)

        # Convert back to PIL format
        result_image = Image.fromarray(resized_red_region)
    else:
        # If no red regions are found, return a blank image
        result_image = Image.new('RGB', output_size, (255, 255, 255))

    return result_image
input_image_path = 'all/RL-Fernwurf_1gg1-WAS_über-Block_1729641574107.png'
original_image = Image.open(input_image_path)

# Apply the red region extraction and randomization
output_image = extract_red_region_and_remove_black(original_image)

# Display the original and processed images side by side for comparison
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(original_image)
# plt.axis('off')
# 
# plt.subplot(1, 2, 2)
# plt.title("Processed Image")
# plt.imshow(output_image)
# plt.axis('off')
# 
# plt.show()