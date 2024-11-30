# Project Name: Quick Notes

## Introduction

Quick Notes is a multifunctional document editing and knowledge management platform designed to help users edit and organize documents more efficiently. With integrated AI features such as text-to-speech, OCR, translation, etc., users can enhance their editing experience and easily organize and present information.

## Language Switching

- [English](./README_EN.md)
- [中文](./README.md)

## Features

### User Management

- **Login**: Users can access the system by logging in.
- **Register**: New users can create an account by registering.
- **Logout**: Users can log out to exit the system.

### Data Management

- **Data Import**: Users can import data for mind maps and flowcharts.
- **Data Export**: Users can export data for mind maps and flowcharts.

### Mind Map and Flowchart

- **Mind Map**: Users can create and edit mind maps.
- **Flowchart**: Users can create and edit flowcharts.
- **Edit**: Provides editing features, allowing users to insert, delete nodes, and set node styles, etc.

### AI Features

- **Text-to-Speech**: Converts text to speech.
- **OCR**: Recognizes text in images.
- **Translation**: Provides multilingual translation features.
- **Writing Assistance**: Provides writing assistance features to help users edit documents better.

### File Management

- **File Upload**: Users can upload files to the system.
- **File Management**: Users can view and manage uploaded files.

### User Profile and Settings

- **Profile**: Users can view and modify their personal information.
- **Settings**: Users can adjust themes and layouts, etc.

### Error Handling and User Feedback

- **Error Handling**: When an error occurs, users can view the error message.
- **User Feedback**: Users can submit feedback to help improve the system.

## Installation

Based on the project structure you provided, here are the detailed installation steps:

1. **Clone the project to your local machine**:
   First, you need to clone the project to your local machine. You can use the following command:
   ```bash
   git clone https://github.com/xiayu1118/Quick-Notes.git
   ```

2. **Install dependencies**:
   The project uses technologies such as Docker, Python (pip), and Node.js (npm). You need to install these dependencies separately.

   - **Docker**:
     Docker can run on Windows, Mac, and Linux. You can download and install Docker from the official Docker website. After installation, you can use the following command in the project root directory to start the middleware deployed in Docker:
     ```bash
     docker-compose up -d
     ```
     This will start Docker containers including mysql, redis, etc.

   - **Python (pip)**:
     You need to install Python and pip. You can download and install Python from the official Python website. After installation, you can use the following command to install the Python dependencies of the project:
     ```bash
     pip install -r requirements.txt
     ```
     This will install all the Python dependencies listed in the `requirements.txt` file.

   - **Node.js (npm)**:
     You need to install Node.js and npm. You can download and install Node.js from the official Node.js website. After installation, you can use the following command to install the Node.js dependencies of the project:
     ```bash
     npm install
     ```
     This will install all the Node.js dependencies listed in the `package.json` file.

3. **Start the project**:
   The project uses technologies such as Docker, Python, and Node.js. You need to start these services separately.
   - **Python**:
     You can use the following command to start the Python application:
     ```bash
     python run.py
     ```
     This will start the Python application and listen on port 5000.

   - **Node.js**:
     You can use the following command to start the Node.js application:
     ```bash
     npm run dev
     ```
     This will start the Node.js application and listen on port 8080.

Now, your project should have been successfully installed and running. You can view the project by visiting `http://localhost:8080`.

## Usage

1. Register or log in to your account.
2. Create or import mind maps and flowcharts.
3. Enhance your editing experience with AI features.
4. Upload and manage files.
5. View and modify your personal information and settings.
6. Submit feedback and error handling.

## Contribution

If you have any suggestions or feedback, please submit an Issue or Pull Request.

## License

This project is licensed under the MIT License.
