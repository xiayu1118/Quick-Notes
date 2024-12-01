from datetime import datetime
from flask_sqlalchemy import SQLAlchemy  

db = SQLAlchemy()  

class User(db.Model):
    __tablename__ = 'user'

    class GENDER:
        MALE = 0
        FEMALE = 1

    id = db.Column('user_id', db.Integer, primary_key=True, doc='用户ID')
    mobile = db.Column(db.String, doc='手机号')
    password = db.Column(db.String, doc='密码')
    name = db.Column('user_name', db.String, doc='昵称')
    gender = db.Column(db.Integer, default=GENDER.FEMALE, doc='性别')
    birthday = db.Column(db.Date, doc='生日')
    is_delete = db.Column(db.Boolean, default=False, doc='是否删除')
    # 当模型类字段与表字段不一致，可在Column函数第一个参数指定
    time = db.Column('create_time', db.DateTime, default=datetime.now, doc='创建时间')
    update_time = db.Column('update_time', db.DateTime, default=datetime.now, onupdate=datetime.now, doc='更新时间')

    # primaryjoin定义连接条件 : param1:另外一方类名 param2: 具体连接条件
    follows = db.relationship('Car', primaryjoin='User.id==foreign(Car.user_id)')
