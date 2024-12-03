from flask import Blueprint
bp = Blueprint('document', __name__, url_prefix='/document')
# 导入数据库模块
import pymysql
# 导入Flask框架，这个框架可以快捷地实现了一个WSGI应用
from apps.config import attap_db, current_time
from flask import jsonify, request

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
api_key="ff18ba62b3fd3251ba7a7601db0992b8.9aXqtH8F6ona5kFb"
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@bp.route('/upload_file', methods=['POST','GET'])
def upload_file():
    try:
        username = request.json.get('user')
        content = request.json.get('content')
        docname = request.json.get('name')
        sorts=request.json.get('sorts')
        conn = pymysql.connect(host=attap_db.get('HOST'), port=attap_db.get('PORT'),
                               user=attap_db.get('USER'), password=attap_db.get('PASSWORD'),
                               charset=attap_db.get('CHARSET'))
        cursor = conn.cursor()
        conn.commit()
        cursor.execute('USE editdata')
        sql = "INSERT INTO document (username, content, docname, times, sort) VALUES (%s, %s, %s, %s, %s)"
        values = (username, content, docname, current_time(),sorts)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"message": "Document saved successfully!"}), 200
    except Exception as e:
        # 捕获并记录错误
        print(f"An error occurred: {str(e)}")
        return jsonify({"message": "Internal Server Error"}), 500
@bp.route('/show_file',methods=['GET', 'POST'])
def show_file():
    conn=None
    account = request.json.get("user")
    conn = pymysql.connect(host=attap_db.get('HOST'), port=attap_db.get('PORT'), user=attap_db.get('USER'),
                           password=attap_db.get('PASSWORD'), charset=attap_db.get('CHARSET'))
    cursor = conn.cursor()
    conn.commit()

    cursor.execute('use editdata')
    if not account:
        return jsonify({'success': 'false', 'message': '用户名未提供'})
    sql = "SELECT * FROM document WHERE username = %s"
    cursor.execute(sql, (account,))
    ducument_data = cursor.fetchall()
    temp_wdnmd=list()
    for i in range (len(ducument_data)):
       a= {'date':ducument_data[i][3],'name':ducument_data[i][2],'sorts':ducument_data[i][5]}
       temp_wdnmd.append(a)

    return jsonify({'success': 'true', 'data':temp_wdnmd})
@bp.route('/get_document',methods=['GET', 'POST'])
def get_document():
    conn=None
    conn = pymysql.connect(host=attap_db.get('HOST'), port=attap_db.get('PORT'),
                           user=attap_db.get('USER'), password=attap_db.get('PASSWORD'),
                           charset=attap_db.get('CHARSET'))
    cursor = conn.cursor()
    conn.commit()
    cursor.execute('USE editdata')
    account=request.json.get('user')
    name = request.json.get('name')
    sql = "SELECT * FROM document WHERE username = %s AND docname = %s"
    cursor.execute(sql, (account,name,))
    ducument_data=list()
    ducument_data = cursor.fetchall()
    return jsonify({'success': 'true', 'content': ducument_data[0][1]})