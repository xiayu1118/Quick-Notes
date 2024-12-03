import re

from flask import Blueprint

bp = Blueprint('AI', __name__, url_prefix='/AI')
# 导入数据库模块
import shutil
import uuid
from apps.mindmap import mindmap_changess
# 导入Flask框架，这个框架可以快捷地实现了一个WSGI应用
from werkzeug.utils import secure_filename
from flask import jsonify, request
import os
import erniebot
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from docx import Document
from apps.milvus import (
    extract_text_from_pdf,
    extract_text_from_image, process_files, generate_text_summaries
)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

api_key = "ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route('/wordtopic', methods=["GET", "POST"])
def wordtopic():
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)  # 请填写您自己的APIKey
    promopt = request.json.get('promopt')
    response = client.images.generations(
        model="cogview-3",  # 填写需要调用的模型名称
        prompt=str(promopt),
    )
    print(response.data[0].url)
    return response.data[0].url


@bp.route('/getAI', methods=["GET", "POST"])
def getAI():
    # 获取用户名
    account = request.form.get("username")
    # 获取用户提问内容
    number = request.form.get("number")
    quescont = request.form.get("cont")
    askcont = ""
    contentss = ""

    # 翻译
    if number == '2':
        language1 = request.form.get("language1")
        language2 = request.form.get("language2")
        askcont = "帮我把下面这段话从" + str(language1) + "翻译为" + str(language2) + ":" + quescont
    # OCR
    elif number == '3':
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400

        if file:
            if not account:
                return jsonify({'error': 'Account not provided'}), 400

            # 创建用户目录
            user_folder = os.path.join("static", "temp_image", account)
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            # 保存文件到用户目录
            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)
            # 调用图片识别函数
            upload_folder_img = r"C:/Users/lenovo/Desktop/others/EditEnd/apps/static/temp_image/"
            upload_folder_img1 = upload_folder_img + str(account) + '/'
            recognized_text = extract_text_from_image(upload_folder_img1)
            recognized_text = str(recognized_text)
            shutil.rmtree(upload_folder_img1)  # 删除整个文件夹及其内容
            os.makedirs(upload_folder_img1)  # 重新创建该文件夹
            return jsonify({'answer': recognized_text})
    # 润色
    elif number == '4':
        format = request.form.get("format")
        tone = request.form.get("tone")
        length = request.form.get("length")
        language3 = request.form.get("language3")
        askcont = "帮我修改这段话" + quescont + "格式要求" + str(format) + "语气要求" + str(tone) + "长度要求" + str(
            length) + "语言要求" + "中文"
    # 一般的gpt问话
    if number == '1':
        # 原始上传文件夹路径
        upload_folder_img = r"C:\Users\lenovo\Desktop\others\EditEnd\apps\static\temp_image"
        askcont = quescont
        file = request.files.get('file')

        if file:
            if file.filename == '':
                return jsonify({'error': 'No file selected for uploading'}), 400

            if not account:
                return jsonify({'error': 'Account not provided'}), 400

            # 创建用户目录
            user_folder = os.path.join(upload_folder_img, account)
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)

            # 保存文件到用户目录
            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)

            # 判断文件类型并处理
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                upload_folder_img1 = os.path.join(upload_folder_img, account)
            else:
                upload_folder_img1 = os.path.join(upload_folder_img, account, filename)
            recognized_text = process_files(upload_folder_img1)

            # 处理recognized_text的内容
            contentss = ""
            for a in recognized_text:
                for b in a:
                    contentss += str(b)

            # 生成摘要文本
            aaaaa = generate_text_summaries(texts=contentss, tables="")
            contentss = ""
            for a in aaaaa:
                for b in a:
                    contentss += b

            # 清理并重新创建文件夹
            shutil.rmtree(user_folder)
            os.makedirs(user_folder)
        else:
            contentss = ""

        if contentss:
            askcont = askcont + "基于以下内容回答：" + contentss

    # 配置 ErnieBot
    erniebot.api_type = 'aistudio'
    erniebot.access_token = '90ba6a5d0e6d72c90ba3f50f997a533659848788'

    try:
        response = erniebot.ChatCompletion.create(
            model='ernie-bot',
            messages=[{'role': 'user', 'content': askcont}],
        )
        restext = response.get_result()
        webdict = {'answer': restext}
        print(webdict)
        return jsonify(webdict)
    except Exception as e:
        print(f"AI处理过程中发生异常: {e}")
        return jsonify({'success': 'false', 'state': 500, 'message': 'AI处理过程中发生异常'})


@bp.route('/mindmap_change', methods=["GET", "POST"])
def mindmap_change():
    descriptionsss = request.json.get("words")
    result = mindmap_changess(descriptionsss)
    pattern = re.compile(r'```json(.*?)```', re.DOTALL)
    matches = pattern.findall(result)
    print(matches)
    matches=str(matches)
    return jsonify({'answer': matches})
