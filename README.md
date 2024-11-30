# 项目名称：匆匆笔记

## 简介

匆匆笔记是一个多功能的文档编辑和知识管理平台，旨在帮助用户更高效地编辑和整理文档。通过集成的AI功能，如文本转语音、OCR、翻译等，用户可以提升编辑体验，更轻松地组织和展示信息。

## 语言切换

- [English](./README_EN.md)
- [中文](./README.md)

## 功能

### 用户管理

- **登录**：用户可以通过登录来访问系统。
- **注册**：新用户可以通过注册来创建账户。
- **登出**：用户可以登出以退出系统。

### 数据管理

- **数据导入**：用户可以导入思维导图和流程图的数据。
- **数据导出**：用户可以导出思维导图和流程图的数据。

### 思维导图和流程图

- **思维导图**：用户可以创建和编辑思维导图。
- **流程图**：用户可以创建和编辑流程图。
- **编辑**：提供编辑功能，用户可以插入、删除节点，设置节点样式等。

### AI功能

- **文本转语音**：将文本转换为语音。
- **OCR**：识别图片中的文字。
- **翻译**：提供多语言翻译功能。
- **写作辅助**：提供写作辅助功能，帮助用户更好地编辑文档。

### 文件管理

- **文件上传**：用户可以上传文件到系统。
- **文件管理**：用户可以查看和管理上传的文件。

### 用户资料和设置

- **个人资料**：用户可以查看和修改自己的个人资料。
- **设置**：用户可以调整主题和布局等。

### 错误处理和用户反馈

- **错误处理**：当发生错误时，用户可以查看错误信息。
- **用户反馈**：用户可以提交反馈，帮助系统改进。

## 安装

根据您提供的项目结构，以下是详细的安装步骤：

1. **克隆项目到本地**：
   首先，您需要将项目克隆到本地。您可以使用以下命令：
   ```bash
   git clone https://github.com/xiayu1118/Quick-Notes.git
   ```

2. **安装依赖**：
   项目使用了Docker、Python（pip）和Node.js（npm）等技术。您需要分别安装这些依赖。

   - **Docker**：
     Docker可以在Windows、Mac和Linux上运行。您可以从Docker官方网站下载并安装Docker。安装完成后，您可以直接在项目根目录下使用以下命令启动部署在Docker的中间件：
     ```bash
     docker-compose up -d
     ```
     这将启动包括mysql，redis等的Docker容器。

   - **Python（pip）**：
     您需要安装Python和pip。您可以从Python官方网站下载并安装Python。安装完成后，您可以使用以下命令安装项目的Python依赖：
     ```bash
     pip install -r requirements.txt
     ```
     这将安装`requirements.txt`文件中列出的所有Python依赖。

   - **Node.js（npm）**：
     您需要安装Node.js和npm。您可以从Node.js官方网站下载并安装Node.js。安装完成后，您可以使用以下命令安装项目的Node.js依赖：
     ```bash
     npm install
     ```
     这将安装`package.json`文件中列出的所有Node.js依赖。

3. **启动项目**：
   项目使用了Docker、Python和Node.js等技术。您需要分别启动这些服务。
   - **Python**：
     您可以使用以下命令启动Python应用程序：
     ```bash
     python run.py
     ```
     这将启动Python应用程序，并监听5000端口。

   - **Node.js**：
     您可以使用以下命令启动Node.js应用程序：
     ```bash
     npm run dev
     ```
     这将启动Node.js应用程序，并监听8080端口。

现在，您的项目应该已经成功安装并运行。您可以通过访问`http://localhost:8080`来查看项目。

## 使用

1. 注册或登录账户。
2. 创建或导入思维导图和流程图。
3. 使用AI功能提升编辑体验。
4. 上传和管理文件。
5. 查看和修改个人资料和设置。
6. 提交反馈和错误处理。

## 贡献

如果您有任何建议或反馈，请提交Issue或Pull Request。

## 许可证

本项目采用MIT许可证。

---

