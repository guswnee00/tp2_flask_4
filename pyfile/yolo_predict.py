import os
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import glob

def yolo(model_path, image_path):
    model = YOLO(model_path)
    image_files = [image_path]

    file_result = []
    
    for index, image_file in enumerate(image_files):
        image_results = {}
        
        results = model(image_file)
        for r in results:
            detected_classes = r.boxes.cls.tolist()  # Convert tensor to a Python list
            detected_locations = r.boxes.xyxy.tolist()  # Convert tensor to a Python list
            for class_label, location in zip(detected_classes, detected_locations):
                if class_label not in image_results:
                    image_results[class_label] = []
                image_results[class_label].append(location)
        
        file_result.append([index, image_results])  # Append index and image_results to file_result list
    return file_result