import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


def remove_watermark(image):
    # 假设水印为灰色，将其转化为掩码
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 100])  # 根据灰色范围调整下限
    upper_gray = np.array([180, 50, 200])  # 根据灰色范围调整上限
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # 用白色覆盖水印区域
    image[mask > 0] = [255, 255, 255]  # 将被水印覆盖的区域设为白色
    return image


def pdf_to_images(pdf_path):
    # 将 PDF 的前 3 页转换为图像
    images = convert_from_path(pdf_path, first_page=1, last_page=3)
    return images


def process_pdf(input_pdf_path, output_pdf_path):
    # 1. 将 PDF 文件的前 3 页转换为图像
    images = pdf_to_images(input_pdf_path)

    processed_images = []

    # 2. 遍历图像并去除水印
    for i, img in enumerate(images):  # 只处理前 3 页
        print(f"Processing page {i + 1}")

        # 将 PIL Image 转换为 OpenCV 格式
        open_cv_image = np.array(img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # 去除水印
        processed_image = remove_watermark(open_cv_image)

        # 将 OpenCV 图像转换回 PIL Image 格式
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(processed_image)

        processed_images.append(pil_image)

    # 3. 将处理后的图像保存为新的 PDF
    if processed_images:
        processed_images[0].save(output_pdf_path, save_all=True, append_images=processed_images[1:])


# 使用正确的路径并确保路径中不存在特殊字符
input_path = r"C:\Users\lenovo\Desktop\物资招标〔2022〕28号　省公司物资类采购策略指导手册（2023版）(1).pdf"
output_path = r"C:\Users\lenovo\Desktop\物资招标_去水印版.pdf"

# 调用函数
process_pdf(input_path, output_path)
