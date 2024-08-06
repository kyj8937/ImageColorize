import requests
import os
from bs4 import BeautifulSoup

# 크롤링할 웹페이지 URL 설정
base_url = "https://www.musinsa.com/mz/streetsnap?_mon=&gender=&p="  # 기본 URL을 설정합니다.
start_page = 1  # 시작 페이지 번호를 설정합니다.
end_page = 35  # 종료 페이지 번호를 설정합니다.

# 이미지 번호 설정
image_number = 0

# 웹페이지 순회
for page_number in range(start_page, end_page + 1):
    url = f"{base_url}{page_number}"  # 현재 페이지의 URL을 생성합니다.

    # 웹페이지에 접속하여 HTML 데이터 가져오기
    response = requests.get(url)
    html = response.text

    # HTML 파싱
    soup = BeautifulSoup(html, "html.parser")

    # 이미지 태그(<img>)를 찾아서 이미지 URL 가져오기
    img_tags = soup.find_all("img")  # 이미지 태그(<img>)를 모두 찾습니다.
    for img_tag in img_tags:
        img_url = img_tag["src"]  # 이미지 태그의 src 속성을 가져옵니다.
        
        # 특정 이미지인 경우 건너뛰고 다음 이미지로 넘어갑니다.
        if img_url == "//image.msscdn.net/skin/musinsa/images/menu_handle.gif":
            continue
        
        if img_url.startswith("//"):
            img_url = "http:" + img_url  # 스키마가 누락된 경우 스키마를 추가합니다.


        # 이미지 다운로드
        save_dir = "Colorimages"  # 이미지를 저장할 디렉토리 경로를 설정합니다.
        os.makedirs(save_dir, exist_ok=True)  # 저장 디렉토리를 생성합니다.

        image_name = f"image_{image_number}.jpg"  # 저장할 이미지 파일명을 설정합니다.
        save_path = os.path.join(save_dir, image_name)  # 이미지 파일의 저장 경로를 설정합니다.

        try:
            response = requests.get(img_url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"다운로드 완료: {image_name}")
        except Exception as e:
            print(f"다운로드 실패: {image_name}, 오류 메시지: {e}")

        image_number += 1  # 이미지 번호를 증가시킵니다.
