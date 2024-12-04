from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from models import db  # 确保从 models 初始化 db
from models import *  # 导入所有模型

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:rootpassword@127.0.0.1:3306/mydatabase'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库和迁移
db.init_app(app)
migrate = Migrate(app, db)

if __name__ == '__main__':
    app.run()
