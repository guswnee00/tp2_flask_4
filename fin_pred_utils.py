from pyfile.seg_predict import *
from pyfile.yolo_predict import *
from pyfile.label_color_list import *
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import cv2
import glob
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box


"""
파일 확장자를 체크하는 함수

    : 허용된 확장자인지 아닌지 확인함
"""
def allowed_file(filename):

    # 허용된 확장자 설정
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


"""
이미지를 전처리하는 함수

    : 이미지를 리사이징해서 그 결과 이미지의 경로를 return값으로 받음 
"""
def resize_with_padding(image_path, 
                        target_width = 960, 
                        target_height = 540):
    
    img = Image.open(image_path)

    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    # 타겟 크기에 맞게 업스케일 또는 다운스케일 계산
    if aspect_ratio > (target_width / target_height):
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height

    # 이미지 크기 조정
    resized_img = img.resize((new_width, new_height))

    # 패딩을 추가하기 위해 빈 캔버스 생성
    padded_img = Image.new("RGB", (target_width, target_height))

    # 패딩 계산
    padding_x = (target_width - new_width) // 2
    padding_y = (target_height - new_height) // 2

    # 조정된 이미지를 패딩을 고려하여 중앙에 배치
    padded_img.paste(resized_img, (padding_x, padding_y))

    return padded_img


"""
segmentation모델과 yolo모델을 합친 함수

    : 각각의 모델에서 나온 결과를 합치는 함수
      최종 이미지를 predictions 디렉토리에 저장
"""
def seg_and_yolo(input_folder,
                 output_folder, 
                 seg_path = "/Users/hyunjulee/tp2/tp2_flask_2/model/UNet_8.pt", 
                 yolo_path = "/Users/hyunjulee/tp2/tp2_flask_2/model/best_1.pt" ):
    
    file_results = yolo(yolo_path, input_folder)
    sample_info = segmen_folder(seg_path, input_folder)

    # yolo의 key와 value를 꺼내기 위한 list 생성 -> 사진은 한장만 있어서 해당 방법으로 파싱
    keys = list(file_results[0][1].keys())
    values = list(file_results[0][1].values())
    image_files = glob.glob(input_folder+ "*.jpg") 

    # seg 중 주요한 라벨 생성
    stop_line = sample_info[1]
    middle_line = sample_info[3]
    crosswalk = sample_info[10]
    intersection = sample_info[6]

    stop_line = np.array(stop_line[0])
    middle_line = np.array(middle_line[0])
    crosswalk = np.array(crosswalk[0])
    intersection = np.array(intersection[0])

    for image_file in image_files:
        image = cv2.imread(image_file)
        for i in range(len(keys)):
            label = keys[i]

            # Load the image using cv2.imread()
            if label == 0 or label == 1 or label == 2 or label == 7 or label == 8 or label == 9 or label == 10 or label == 18 or label == 16 or label == 17 or label == 18:
                color = yolo_color.get(label)
                # Draw bounding box
                cv2.rectangle(image, (int(values[i][0][0]), int(values[i][0][1])), (int(values[i][0][2]), int(values[i][0][3])), color, 2)
                
                # Put label text
                class_korea_label = yolo_label[label]
                cv2.putText(image, class_korea_label, (int(values[i][0][0]), int(values[i][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            elif label == 3 or label == 4 or label == 5 or label == 6:
                rectangle_coords = [(values[i][0][0], values[i][0][1]), (values[i][0][2], values[i][0][1]), (values[i][0][2], values[i][0][3]), (values[i][0][0], values[i][0][3])]
                if middle_line.any():
                    rectangle = Polygon(rectangle_coords)
                    polygon = Polygon(middle_line)
                    if rectangle.intersects(polygon):
                        label = 3
                    else:
                        pass
                
                elif stop_line.any():
                    rectangle = Polygon(rectangle_coords)
                    polygon = Polygon(middle_line)
                    if rectangle.intersects(polygon):
                        label = 4
                    else:
                        pass
                    
                elif crosswalk.any():
                    rectangle = Polygon(rectangle_coords)
                    polygon = Polygon(middle_line)
                    if rectangle.intersects(polygon):
                        label = 5
                    else:
                        pass
                    
                elif intersection.any():
                    rectangle = Polygon(rectangle_coords)
                    polygon = Polygon(middle_line)
                    if rectangle.intersects(polygon):
                        label = 6
                    else:
                        label = label
                

                color = yolo_color.get(label)
                # Draw bounding box
                cv2.rectangle(image, (int(values[i][0][0]), int(values[i][0][1])), (int(values[i][0][2]), int(values[i][0][3])), color, 2)
                
                # Put label text
                class_korea_label = yolo_label[label]
                cv2.putText(image, class_korea_label, (int(values[i][0][0]), int(values[i][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            else:
                color = yolo_color.get(label)
                # Draw bounding box
                cv2.rectangle(image, (int(values[i][0][0]), int(values[i][0][1])), (int(values[i][0][2]), int(values[i][0][3])), color, 2)
                
                # Put label text
                class_korea_label = yolo_label[label]
                cv2.putText(image, class_korea_label, (int(values[i][0][0]), int(values[i][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        original_image_name, _ = os.path.splitext(os.path.basename(image_file))
        new_image_path = output_folder + f'pred_{original_image_name}.jpg'
        cv2.imwrite(new_image_path, image)
        
        return new_image_path
       
        
# 확인 
#print(seg_and_yolo(input_folder = '/Users/hyunjulee/tp2/tp2_flask_2/static/temps/',
#                   output_folder = '/Users/hyunjulee/tp2/tp2_flask_2/static/predictions/'))


