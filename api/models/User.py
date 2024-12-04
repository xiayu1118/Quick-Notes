from . import db  # 从 models 包中引入 db   
class User(db.Model):  
    __tablename__ = 'user'  
    
    id = db.Column(db.Integer, primary_key=True)  
    account = db.Column(db.String(50), unique=True, nullable=False)  
    password = db.Column(db.String(128), nullable=False)  
    
    # 方法可以根据需要添加  
    def __repr__(self):  
        return f'<User {self.account}>'  
    




    

class Person(db.Model):  
    __tablename__ = 'person'  
    
    id = db.Column(db.Integer, primary_key=True)  
    account = db.Column(db.String(50), db.ForeignKey('user.account'), nullable=False)  
    email = db.Column(db.String(100), nullable=True)  
    phone = db.Column(db.String(20), nullable=True)  
    token = db.Column(db.String(100), nullable=True)  
    pet_name = db.Column(db.String(50), nullable=True)  
    
    user = db.relationship('User', backref=db.backref('persons', lazy=True))  

    def __repr__(self):  
        return f'<Person {self.account}>'  

class ImgPath(db.Model):  
    __tablename__ = 'imgpath'  
    
    id = db.Column(db.Integer, primary_key=True)  
    username = db.Column(db.String(50), db.ForeignKey('user.account'), nullable=False)  
    url = db.Column(db.String(200), nullable=False)  
    picname = db.Column(db.String(100), nullable=False)  

    user = db.relationship('User', backref=db.backref('img_paths', lazy=True))  

    def __repr__(self):  
        return f'<ImgPath {self.picname} for {self.username}>'