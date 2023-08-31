from PIL import Image
import os
import shutil

def resize_with_padding(image_path, target_width, target_height):
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

upload_folder = r"C:\Users\antae\Desktop\Upload"  # 업로드된 이미지가 위치한 폴더
test_folder = r"C:\Users\antae\Desktop\Test"      # 결과 이미지를 저장할 폴더

# 업로드 폴더 내 수정 시간을 기준으로 파일 정렬하여 최근 파일 선택
uploaded_images = [f for f in os.listdir(upload_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
uploaded_images.sort(key=lambda x: os.path.getmtime(os.path.join(upload_folder, x)), reverse=True)

if uploaded_images:
    uploaded_image_path = os.path.join(upload_folder, uploaded_images[0])  # 최근 수정된 이미지 사용
    output_image_path = os.path.join(test_folder, "1.jpg")  # 결과 이미지 경로

    output_image = resize_with_padding(uploaded_image_path, 960, 540)
    output_image.save(output_image_path)
