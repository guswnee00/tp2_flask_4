import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from pyfile.model import *
from pyfile.data import *
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2

def segmen_folder(model_path, input_folder):
    # Define color map for class labels
    color_map = {
        1: (34, 193, 195),     # Red
        2: (201, 63, 32),     # Green
        3: (59, 47, 127),   # Purple
        4: (136, 185, 32),   # Magenta
        5: (222, 194, 197),   # Teal
        6: (71, 61, 40),   # Olive
        7: (207, 73, 210),     # Blue
        8: (137, 121, 131),   # Yellow
        9: (62, 67, 218),     # Maroon
        10: (128, 203, 202),    # Lime
        11: (45, 191, 68),   # Indigo
        12: (206, 207, 85), # Gray
        13: (207, 52, 125) #Black
    }
    # Load your model here and set it to evaluation mode
    model_path = model_path
    model = UNet(num_classes=13)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Define the DataLoader and other necessary variables
    input_folder = input_folder
    test_batch_size = 1

    dataset = CityscapeDataset_predict(input_folder, label_model)
    data_loader = DataLoader(dataset, batch_size=test_batch_size)

    # Process each batch and save segmentation results
    for batch_idx, X in enumerate(data_loader):
        Y_pred = model(X)

        # Convert predicted classes to color image using color_map
        color_image = np.zeros((Y_pred.shape[2], Y_pred.shape[3], 3), dtype=np.uint8)
        for class_idx in range(1, Y_pred.shape[1] + 1):  # Class indices are 1-based in color_map
            class_mask = (Y_pred[0].argmax(dim=0) == class_idx)  # Get mask for current class
            color = color_map[class_idx]
            color_image[class_mask] = color
        
        # Process each class label and find polygons
        polygons_by_class = {}

        for class_idx in color_map.keys():
            if class_idx == 8:  # Skip label 8
                continue
            
            class_mask = (color_image == color_map[class_idx]).all(axis=2)  # Get mask for current class
            contours = find_contours(class_mask, 0.5)
            
            class_polygons = []
            for contour in contours:
                # Convert contour to polygon vertices
                contour = np.flip(contour, axis=1)  # Convert (row, col) to (x, y) format
                polygon = contour.tolist()  # Convert NumPy array to nested Python list
                class_polygons.append(polygon)
            
            polygons_by_class[class_idx] = class_polygons
    return polygons_by_class

# def segmen_image(model_path, input_image, batch_size):
#     # Define color map for class labels
#     color_map = {
#         1: (34, 193, 195),     # Red
#         2: (201, 63, 32),     # Green
#         3: (59, 47, 127),   # Purple
#         4: (136, 185, 32),   # Magenta
#         5: (222, 194, 197),   # Teal
#         6: (71, 61, 40),   # Olive
#         7: (207, 73, 210),     # Blue
#         8: (137, 121, 131),   # Yellow
#         9: (62, 67, 218),     # Maroon
#         10: (128, 203, 202),    # Lime
#         11: (45, 191, 68),   # Indigo
#         12: (206, 207, 85), # Gray
#         13: (207, 52, 125) #Black
#     }
#     # Load your model here and set it to evaluation mode
#     model_path = model_path
#     model = UNet(num_classes=13)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     model.eval()

#     # Define the DataLoader and other necessary variables
#     input_folder = np.array(input_image)
#     test_batch_size = batch_size

#     dataset = CityscapeDataset_image(input_folder, label_model)
#     data_loader = DataLoader(dataset, batch_size=test_batch_size)

#     # Process each batch and save segmentation results
#     for batch_idx, X in enumerate(data_loader):
#         Y_pred = model(X)

#         # Convert predicted classes to color image using color_map
#         color_image = np.zeros((Y_pred.shape[2], Y_pred.shape[3], 3), dtype=np.uint8)
#         for class_idx in range(1, Y_pred.shape[1] + 1):  # Class indices are 1-based in color_map
#             class_mask = (Y_pred[0].argmax(dim=0) == class_idx)  # Get mask for current class
#             color = color_map[class_idx]
#             color_image[class_mask] = color
        
#         # Process each class label and find polygons
#         polygons_by_class = {}

#         for class_idx in color_map.keys():
#             if class_idx == 8:  # Skip label 8
#                 continue
            
#             class_mask = (color_image == color_map[class_idx]).all(axis=2)  # Get mask for current class
#             contours = find_contours(class_mask, 0.5)
            
#             class_polygons = []
#             for contour in contours:
#                 # Convert contour to polygon vertices
#                 contour = np.flip(contour, axis=1)  # Convert (row, col) to (x, y) format
#                 polygon = contour.tolist()  # Convert NumPy array to nested Python list
#                 class_polygons.append(polygon)
            
#             polygons_by_class[class_idx] = class_polygons
#         return polygons_by_class

# # example
#model_path =  'D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\UNet_8.pt'
#input_folder = 'D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\test_data\\train\\pre_testing\\train\\1image'
#sample_list = segmen_folder(model_path, input_folder)
#print(sample_list.keys())
   
# # Load an example image
# image_path = 'D:\\tp2\\Teamproject2\\image-segmentation-yolov8\\test_data\\train\\pre_testing\\train\\1image\\C000002_003_0009_C_D_F_0.jpg'
# image = Image.open(image_path)

# # Convert the image to NumPy array
# image_np = np.array(image)

# # Create an empty mask with the same shape as the image
# mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

# # Draw polygons on the mask
# for class_idx, polygons in polygons_by_class.items():
#     for polygon in polygons:
#         polygon_np = np.array(polygon, dtype=np.int32)
#         cv2.fillPoly(mask, [polygon_np], (255))

# # Apply the mask to the image
# masked_image = cv2.bitwise_and(image_np, image_np, mask=mask)

# plt.imshow(masked_image)
# plt.axis('off')  # Turn off axis labels and ticks
# plt.show()