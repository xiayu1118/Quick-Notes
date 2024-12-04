import subprocess
import os

# 设置环境变量
os.environ["XINFERENCE_MODEL_SRC"] = "modelscope"

# 定义命令
command = ["xinference-local", "--host", "0.0.0.0", "--port", "9997"]

try:
    # 执行命令
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"命令执行失败: {e}")
