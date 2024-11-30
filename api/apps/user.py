import cv2
import pymysql
# 导入Flask框架，这个框架可以快捷地实现了一个WSGI应用
from flask import json
from docx import Document
from apps.config import attap_db
import json
from flask import Flask, jsonify, request
import os

import numpy as np
from flask_cors import CORS
#蓝图操作
from apps.multimodels import bp as multimodel_bp
from apps.AI import bp as AI_bp
from apps.document import bp as document_bp
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, resource={r'/*': {'origins': '*'}})

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb"
#注册蓝图
app.register_blueprint(multimodel_bp)
app.register_blueprint(document_bp)
app.register_blueprint(AI_bp)
app.secret_key = os.urandom(24).hex()
app.config.from_object(__name__)
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
app.config.from_object(__name__)
CORS(app, resource={r'/*': {'origins': '*'}})
app.config['JSON_AS_ASCII'] = False
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb"
#注册
@app.route('/registuser', methods=['POST','GET'])
def get_register_request():
    try:
        # 获取前端传递的用户名和密码
        account = request.json.get("user")
        password = request.json.get("password")
        password2 = request.json.get("password2")
        if password == password2:
            with pymysql.connect(host=attap_db.get('HOST'), port=attap_db.get('PORT'), user=attap_db.get('USER'),
                                password=attap_db.get('PASSWORD'), charset=attap_db.get('CHARSET')) as conn:
                with conn.cursor() as cursor:
                    cursor.execute('use editdata')
                    conn.commit()
                    # 检查用户是否已存在
                    sql = "SELECT * FROM `user` WHERE `account`= %s;"
                    cursor.execute(sql, (account,))
                    result = cursor.fetchone()
                    if result:
                        return jsonify({'success': 'false', 'state': 201, 'message': '你个sb的账号已经有了','content':'{"access_token": ""}'})

                    # 插入新用户
                    # 插入数据到user表
                    sql2 = "INSERT INTO user(account, password) VALUES(%s, %s);"
                    cursor.execute(sql2, (account, password))


                    # 插入数据到person表
                    sql3 = "INSERT INTO `person`(`account`, `email`, `phone`,  `token`, `pet_name`,`password`) VALUES(%s, %s, %s, %s, %s, %s);"
                    cursor.execute(sql3, (
                    account, "example@example.com", "1234567890", "12345678910", "李昌霖的马si了", password))

                    # 提交事务
                    conn.commit()

        else:
            return jsonify({'success': 'false', 'state': 500, 'message': '两次密码不一致，注册失败', 'content':'{"access_token": ""}'})

        # 注册成功，返回JSON响应
        return jsonify({'success': 'true', 'state': 200, 'message': '注册成功', 'content':'{"access_token": ""}'})

    except Exception as e:
        # 发生异常时返回错误信息
        return jsonify({'success': 'false', 'state': 500, 'message': '发生异常，注册失败', 'content':'{"access_token": ""}'})
app.route('/logout', methods=['post'])
#登出
def get_logout_request():
    return jsonify(
        {'success': 'true', 'state': 500, 'message': '登出', 'content': '{"access_token": ""}'})
#登录
@app.route('/login', methods=['post'])
def get_login_request():
    conn = None
    print(1)
    try:
        # 获取前端传递的用户名和密码
        account = request.json.get('user')
        password = request.json.get('password')
        # 连接数据库

        conn = pymysql.connect(host=attap_db.get('HOST'), port=attap_db.get('PORT'), user=attap_db.get('USER'), password=attap_db.get('PASSWORD'), charset=attap_db.get('CHARSET'))
        cursor = conn.cursor()
        conn.commit()
        cursor.execute('use editdata')

        # 使用参数化查询来防止SQL注入

        sql = "SELECT * FROM `user` WHERE `account` = '" + str(account) + "' AND `password` = '" + str(password) + "';"
        res1 = cursor.execute(sql)
        conn.commit()
        if(res1==0):
            return jsonify({'success': 'false', 'state': 500, 'message': 'sb东西密码不对', 'content':'{"access_token": ""}'
            })
        results = cursor.fetchall()
        inner_json = {'access_token': account}
        # 将内部JSON对象转换为字符串
        content_str = json.dumps(inner_json)
        # 构造最终的响应数据
        response_data = {
            'success': 'true',
            'state': 200,
            'message': '对了',
            'content': content_str  # 使用转换后的字符串作为content的值
        }
        # 使用 jsonify 返回 JSON 响应
        return jsonify(response_data)
    except Exception as e:
        # 发生异常时返回错误信息
        return jsonify({'success': 'false', 'state': 500, 'message': '发生异常，登录失败','content':'{"access_token": ""}'})
    finally:
        if conn is not None:
            conn.close()
#个人主页显示
# 上传接口
@app.route('/upload_avatar', methods=['POST'])
def upload_avatar():
    conn = None
    try:
        conn = pymysql.connect(host=attap_db.get('HOST'), port=attap_db.get('PORT'), user=attap_db.get('USER'),
                               password=attap_db.get('PASSWORD'), charset=attap_db.get('CHARSET'))
        cursor = conn.cursor()
        conn.commit()
        cursor.execute('use editdata')
        if 'avatar' not in request.files:
            return jsonify(success='false', message='No file part')
        img = request.files['avatar']
        if img.filename == '':
            return jsonify(success='false', message='No selected file')
        picname=img.filename
        if img and allowed_file(img.filename):
            account = request.form.get('user')
            if not account:
                return jsonify(success='false', message='Account not provided')
            file=img.read()
            file = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)  # 解码为ndarray
            imgfile1_path = "./static/images/" + account + "/"
            if not os.path.exists(imgfile1_path):
                os.makedirs(imgfile1_path)
            # 从前端传来的用户账号·
            img1_path = os.path.join(imgfile1_path, picname)
            cv2.imwrite(filename=img1_path, img=file)
            base_url = request.host_url  # e.g., "http://localhost:5000/"
            file_url = os.path.join(base_url, 'static','images',account,picname).replace("\\", "/")
            url=img1_path
            urls=file_url
            print(urls)
            sql = """
            INSERT INTO imgpath (username, url, picname) 
            VALUES (%s, %s, %s) 
            ON DUPLICATE KEY UPDATE 
                url = VALUES(url), 
                picname = VALUES(picname);
            """
            cursor.execute(sql, (account, url, picname))
            conn.commit()

            return jsonify(success='true', avatar_url=urls)

        else:
            return jsonify(success='false', message='File type not allowed')
    except Exception as e:
        return jsonify(success='false', state=500, message='发生异常', error=str(e))
    finally:
        if conn is not None:
            conn.close()


@app.route('/personal_page', methods=['GET', 'POST'])
def personal_page():
    conn = None
    try:
        # 创建数据库连接
        conn = pymysql.connect(
            host=attap_db.get('HOST'),
            port=attap_db.get('PORT'),
            user=attap_db.get('USER'),
            password=attap_db.get('PASSWORD'),
            charset=attap_db.get('CHARSET')
        )
        cursor = conn.cursor()
        cursor.execute('use editdata')
        conn.commit()
        aaaaa=1

        if aaaaa==1:
            # 获取前端传递的用户名
            account = request.json.get('user')

            if account is None:
                return jsonify({'success': 'false', 'message': '用户名未提供'})

            # SQL查询语句，使用%s作为参数占位符
            sql = "SELECT * FROM person WHERE account = %s"
            cursor.execute(sql, (account,))
            res1 = cursor.fetchall()
            conn.commit()
            result_list = [dict((cursor.description[i][0], value) for i, value in enumerate(row)) for row in res1]
            conn=None
            sql1 = "SELECT * FROM imgpath WHERE username = %s"
            cursor.execute(sql1, (account,))
            res2 = cursor.fetchall()

            if not res2:
                return jsonify({'success': 'false', 'message': '图片路径未找到'})
            picname = res2[0][2]  # Assuming the picture name is in the third column of imgpath table
            base_url = request.host_url  # e.g., "http://localhost:5000/"
            file_url = os.path.join(base_url, 'static', 'images', account, picname).replace("\\", "/")
            # Convert result from database to dictionary
            return jsonify({'success': 'true', 'data': result_list, 'url': file_url})

    except Exception as e:
        return jsonify({'success': 'false', 'state': 500, 'message': '发生异常', 'error': str(e)})
    finally:
        if conn is not None:
            conn.close()


@app.route('/person_page_show', methods=['POST', 'GET'])
def person_page_show():
    conn = None
    try:
        conn = pymysql.connect(host=attap_db.get('HOST'), port=attap_db.get('PORT'), user=attap_db.get('USER'),
                               password=attap_db.get('PASSWORD'), charset=attap_db.get('CHARSET'))
        cursor = conn.cursor()
        conn.commit()
        cursor.execute('use editdata')

        if request.method == 'POST':
            # 获取前端传递的用户名及其他信息
            account = request.json.get('account')
            email = request.json.get('email')
            phone = request.json.get('phone')
            token = request.json.get('token')
            pet_name = request.json.get('pet_name')

            if not account:
                return jsonify({'success': 'false', 'message': '用户名未提供'})
            sql = "UPDATE person SET email = %s, phone = %s, token = %s, pet_name = %s WHERE account = %s"
            cursor.execute(sql, (email, phone, token, pet_name, account))
            conn.commit()
            # 获取更新后的数据
            sql = "SELECT * FROM person WHERE account = %s"
            cursor.execute(sql, (account,))
            updated_user = cursor.fetchall()
            result_list = [dict((cursor.description[i][0], value) for i, value in enumerate(row)) for row in updated_user]
            return jsonify({'success': 'true', 'data': result_list})
    except Exception as e:
        return jsonify({'success': 'false', 'state': 500, 'message': '发生异常', 'error': str(e)})
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    app.run()
