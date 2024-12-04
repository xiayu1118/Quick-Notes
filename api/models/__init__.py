from flask_sqlalchemy import SQLAlchemy

# 初始化数据库对象
db = SQLAlchemy()

# 导入模型
from .user import User  # 确保所有模型类都被导入

# 确保 db 和模型都被正确导出
__all__ = ['db', 'User']
