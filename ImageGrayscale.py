import cv2
import os
from PIL import Image

def convert_to_grayscale(image_path, output_path):
    image = Image.open(image_path).convert('L')  # 이미지를 흑백으로 변환합니다.
    image.save(output_path)

# 이미지가 저장된 폴더 경로
image_folder = 'Colorimages/'

# 변환된 이미지를 저장할 폴더 경로
output_folder = 'grayscale_images/'

# 변환된 이미지를 저장할 폴더가 없다면 생성합니다.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 폴더 내의 모든 이미지 파일에 대해 변환 작업을 수행합니다.
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        output_path = os.path.join(output_folder, filename)
        convert_to_grayscale(image_path, output_path)

print('이미지 변환이 완료되었습니다.')
