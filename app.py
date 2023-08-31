"""
flask app.py

    : 메인 페이지에서 사용자가 이미지를 업로드하면 
      정확히 업로드 되었는지, 허용된 확장자인지 확인 후
      segmentation model과 object detection model이 결합된 seg_and_yolo을 통해서 이미지 예측
"""

import os
from flask import Flask, render_template, request, flash
from fin_pred_utils import *

# 플라스크 클래스명 지정
app = Flask(__name__)

# 에러 페이지
@app.errorhandler(404)
def image_upload_retry(error):
    return render_template('error.html'), 404


# 시작 페이지
@app.route('/')
def home():
    return render_template('home.html')


# 메인 페이지
@app.route('/main', methods=['GET', 'POST'])
def main():
    # 메인 페이지 처음 열린 경우
    if request.method == 'GET':
        return render_template('main.html')
    
    # 사용자가 이미지를 업로드한 경우
    if request.method == 'POST': 
        if 'image' in request.files:
            image = request.files['image']

            # 파일이 업로드되지 않은 경우 
            if image.filename == '':
                flash('No selected file. Please choose an image to upload.')
                return render_template('error.html')
            
            # 확장자가 허용된 확장자인지 확인
            if not allowed_file(image.filename):
                flash('Invalid file. Please upload a valid image file (jpg, jpeg, png, gif).')
                return render_template('error.html')
            
            # 사용자가 업로드한 이미지 저장할 경로 설정 -> 메모리에서 읽어오는 대신 파일로 저장하는 것이 좋음
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok = True)     # uploads 디렉터리가 없다면 생성

            # 리사이징한 이미지 저장할 경로 설정
            temp_folder= os.path.join('static', 'temps')
            os.makedirs(temp_folder, exist_ok = True)    # temps 디렉터리가 없다면 생성 

            # 예측한 이미지 저장할 경로 설정
            pred_folder = os.path.join('static', 'predictions')
            os.makedirs(pred_folder, exist_ok = True)    # temps 디렉터리가 없다면 생성 

            # 이전 이미지 파일 삭제 (upload_folder, temp_folder, pred_folder 내의 이미지를 모두 삭제)
            for folder in [upload_folder, temp_folder, pred_folder]:
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)  # 파일 삭제
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                        
            # 업로드한 이미지 저장
            upload_image_path = os.path.join(upload_folder, image.filename)
            image.save(upload_image_path)

            # 사용자가 업로드한 이미지 이름을 리사이징한 이미지에도 동일하게 적용
            upload_image_name = image.filename 

            # resize_with_padding함수를 이용해 이미지 리사이징  
            padding_image = resize_with_padding(image_path = upload_image_path)

            # 리사이징한 이미지 저장
            padding_image_path = os.path.join(temp_folder, upload_image_name)
            padding_image.save(padding_image_path)
            

            # seg_and_yolo(결합된 모델)를 통해 나온 예측 이미지 경로
            pred_image_path = seg_and_yolo(input_folder = temp_folder + '/', output_folder = pred_folder + '/')
            

            # 이미지를 페이지에 띄워주기 위해 이미지 경로와 예측 결과를 템플릿으로 전달
            return render_template('result.html', image_path=upload_image_path, predictions=pred_image_path)

# 결과 페이지
@app.route('/result')
def result():
    return render_template('result.html')


# 웹 앱 실행
if __name__ == '__main__':
    app.run(debug=True)
