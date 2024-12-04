import os
import string
from datetime import datetime
from random import random

import cv2
import numpy as np
import pymysql
from flask import jsonify, request
from werkzeug.utils import send_from_directory, secure_filename
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def generate_random_string(length):
    # 定义所有可能的字符
    letters_and_digits = string.ascii_letters + string.digits
    # 从所有可能的字符中随机选择指定长度的字符
    random_string = ''.join(random.choice(letters_and_digits) for i in range(length))
    return random_string

attap_db={
    'HOST':'127.0.0.1',
    'PORT':3306,
    'USER':'root',
    'PASSWORD':'root',
    'CAHRSET':'utf-8',
    'NAME':'editdata'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def uploadimages(username, img):
    try:
        picname = secure_filename(img.filename)
        file = img.read()
        # 检查文件是否为空
        print("File read successfully, length:", len(file))

        # Decode the image
        np_arr = np.frombuffer(file, np.uint8)

        # 检查字节数组是否为空
        if np_arr.size == 0:
            raise ValueError("Error: The image byte array is empty.")

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 检查解码后的文件是否为空
        if img is None:
            raise ValueError("Error: Could not decode the image.")

        print("Image decoded successfully")

        imgfile1_path = os.path.join("static/images/", username)
        if not os.path.exists(imgfile1_path):
            os.makedirs(imgfile1_path)

        img1_path = os.path.join(imgfile1_path, picname)
        cv2.imwrite(filename=img1_path, img=img)
        print("Image saved successfully at:", img1_path)

        # OCR 处理
        result = ocr.ocr(img1_path, cls=True)

        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)

        # OCR 处理完成后删除文件
        if os.path.exists(img1_path):
            os.remove(img1_path)
            print("Image file removed after OCR processing")

        return result

    except ValueError as ve:
        print(f"ValueError occurred: {ve}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None 