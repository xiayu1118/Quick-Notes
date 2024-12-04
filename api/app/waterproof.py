import os
import cv2
import fitz  # PyMuPDF
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


def convert_mm_to_pixel(mm, dpi=72):
    """将毫米转换为像素。"""
    return int((mm / 25.4) * dpi)  # 1英寸 = 25.4毫米

from concurrent.futures import ThreadPoolExecutor
import time
def remove_watermark(image):
    """去除图像中的水印，假设水印为灰色区域。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    image[np.where((gray >= 200) & (gray < 240))] = [255, 255, 255]
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    repaired_image = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)
    return repaired_image


def process_page(page, page_num, pic_dir):
    """处理单个页面，提取图像并去除水印。"""
    zoom_x, zoom_y = 1.5, 1.5
    trans = fitz.Matrix(zoom_x, zoom_y)
    pixmap = page.get_pixmap(matrix=trans, alpha=False)
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n).copy()
    if pixmap.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = remove_watermark(img)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_path = os.path.join(pic_dir, f"pdf_split_{page_num + 1}.png")
    pil_img.save(img_path)
    return img_path


def process_pdf(input_file_path, output_file_path):
    """处理整个PDF文件，提取每一页并去除水印。"""
    pic_dir = r"C:\Users\lenovo\Desktop\others\EditEnd\apps\temp_images"
    os.makedirs(pic_dir, exist_ok=True)
    pdf = fitz.open(input_file_path)
    img_paths = [None] * len(pdf)

    # 使用线程池并行处理每一页
    with ThreadPoolExecutor() as executor:
        futures = []
        for page_num, page in enumerate(pdf):
            futures.append(executor.submit(process_page, page, page_num, pic_dir))
        for page_num, future in enumerate(futures):
            img_paths[page_num] = future.result()

    pdf.close()

    # 创建新的 PDF 文件
    pdf_output = fitz.open()
    for img_path in img_paths:
        with Image.open(img_path) as img:
            width, height = img.size
            page = pdf_output.new_page(width=width, height=height)
            # 插入图像到页面上，fitz.Rect(0, 0, width, height) 定义插入区域
            page.insert_image(fitz.Rect(0, 0, width, height), filename=img_path)

    # 保存新的 PDF 文件
    pdf_output.save(output_file_path)
    pdf_output.close()
    # 清理临时文件
    for img_file in img_paths:
        if os.path.exists(img_file):
            try:
                os.remove(img_file)
            except PermissionError:
                print(f"无法删除文件 {img_file}，文件正在使用中。将稍后重试。")
                time.sleep(0.5)  # 尝试延长等待时间
                try:
                    os.remove(img_file)
                except Exception as e:
                    print(f"删除文件 {img_file} 失败：{e}")

    try:
        os.rmdir(pic_dir)
    except OSError:
        print(f"无法删除文件夹 {pic_dir}，可能它不为空。")

# 调用示例
# process_pdf('input.pdf', 'output.pdf')

def remove_watermark_from_image(image):
    """
    去除图像中的水印，假设水印为灰色区域。

    参数:
    image (numpy.ndarray): 输入的图像，OpenCV 格式 (BGR)。

    返回:
    numpy.ndarray: 处理后的图像。
    """
    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 假设水印为灰色，设置灰色的上下限
    lower_gray = np.array([0, 0, 150])  # 灰色下限
    upper_gray = np.array([180, 50, 200])  # 灰色上限

    # 生成掩码，找到灰色区域
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # 将灰色水印区域替换为白色
    image[mask > 0] = [255, 255, 255]

    return image

def process_image1(input_image_path):
    """
    去除给定路径图片的水印并替换为无水印版本。

    参数:
    input_image_path (str): 输入图片的文件路径。

    返回:
    None: 处理完成后覆盖原图片文件。
    """
    # 检查图片文件是否存在
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"文件 {input_image_path} 不存在。")

    # 读取图像
    image = cv2.imread(input_image_path)

    if image is None:
        raise ValueError(f"无法读取文件 {input_image_path}，请检查文件格式。")

    # 去除水印
    processed_image = remove_watermark_from_image(image)

    # 将处理后的图像保存（覆盖原来的图像）
    cv2.imwrite(input_image_path, processed_image)

from PyPDF2 import PdfReader
from decimal import Decimal

def get_pdf_page_dimensions(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        # 读取第一页
        first_page = reader.pages[0]
        # 获取页面尺寸（假设以点为单位）
        width = first_page.mediabox.width
        height = first_page.mediabox.height
        # 将点转换为毫米，确保进行类型转换
        width_mm = Decimal(width) * Decimal(0.352778)  # 将 float 转换为 Decimal
        height_mm = Decimal(height) * Decimal(0.352778)  # 将 float 转换为 Decimal
        return float(width_mm), float(height_mm)  # 转换回 float

# 示例调用
import os
from PIL import Image
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import shutil

def extract_text_from_image(image_path):
    """
    使用 OCR 对图像进行文本识别。
    :param image_path: 图像文件路径。
    :return: 识别出的文本。
    """
    try:
        # 使用 with 确保图像文件正确关闭
        with Image.open(image_path) as img:
            text = pytesseract.image_to_string(img, lang='chi_sim')  # 识别简体中文
        return text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def pdf_to_images(pdf_path, output_dir):
    """
    将 PDF 的每一页转换为图像并保存。
    :param pdf_path: 输入的 PDF 文件路径。
    :param output_dir: 存放图像的目录。
    :return: 生成的图像路径列表。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_path = os.path.join(output_dir, f"page_{page_num}.png")
        pix.save(img_path)
        image_paths.append(img_path)

    doc.close()
    return image_paths

def process_image(img_path):
    """
    多线程调用的单页图像 OCR 处理。
    :param img_path: 图像文件路径。
    :return: (页面编号, 识别出的文本)
    """
    text = extract_text_from_image(img_path)
    page_num = int(os.path.basename(img_path).split("_")[1].split(".")[0])
    return page_num, text

def process_pdf_and_extract_text(input_file_path, output_file_path):
    """
    处理 PDF 文件（去除水印等），并通过 OCR 提取每页的文本。
    :param input_file_path: 输入的 PDF 文件路径。
    :param output_file_path: 输出的处理后的 PDF 文件路径。
    :return: 保存每页文本的列表，按页码顺序排列。
    """
    text_parts = []
    image_dir = "temp_images"
    img_paths = []

    try:
        img_paths = pdf_to_images(output_file_path, image_dir)

        # 使用多线程进行 OCR 处理
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_image, img_path): img_path for img_path in img_paths}

            # 收集结果并按页码排序
            results = []
            for future in as_completed(futures):
                try:
                    page_num, text = future.result()
                    if text:
                        results.append((page_num, text))
                except Exception as e:
                    print(f"Error processing image {futures[future]}: {e}")

            # 等待所有线程完成
            executor.shutdown(wait=True)

            # 按页码顺序排序
            results.sort(key=lambda x: x[0])
            text_parts = [text for _, text in results]

    except Exception as e:
        print(f"Error processing PDF: {e}")

    finally:
        # 清理图像文件和目录
        for img_path in img_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)

    return text_parts
