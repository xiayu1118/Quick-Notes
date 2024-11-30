<template>
  <div class="chat-app">
    <div class="sidebar">
      <h3>知识库列表</h3>
      <ul>
        <li
          v-for="item in knowledgeItems"
          :key="item.id"
          :class="{ selected: selectedItems.includes(item.id) }"
          @click="toggleSelectItem(item.id)"
        >
          {{ item.name }}
        </li>
      </ul>
      <h3>招标文书助手</h3>
      <el-button @click="handleAddTenderDoc">添加招标文书</el-button>

      <!-- 文件上传弹窗 -->
      <el-dialog v-model="uploadDialogVisible" title="上传招标文书" width="30%">
        <el-upload
          action="http://127.0.0.1:5000/excel_match/insert_data"
          name="files"
          accept=".xlsx, .xls"
          :on-success="handleUploadSuccess"
          :on-error="handleUploadError"
          :show-file-list="false"
        >
          <el-button type="primary">点击上传文件</el-button>
        </el-upload>
        <span slot="footer" class="dialog-footer">
          <el-button @click="uploadDialogVisible = false">关闭</el-button>
        </span>
      </el-dialog>

      <!-- 删除文件按钮放在弹窗外 -->
      <el-button type="danger" @click="deleteFile" class="delete-file-btn">删除文件</el-button>

      <div class="tender-doc-helper">
        <el-button
          :class="{'is-active': isAskingTenderQuestion}"
          @click="toggleTenderQuestionMode"
        >
          基于招标文件提问
        </el-button>
      </div>
    </div>

    <div class="history">
      <h3>聊天历史记录</h3>
      <ul>
        <li v-for="history in chatHistory" :key="history.id" @click="toggleSelectchat(history)">
          聊天历史记录ID： {{ history }}
        </li>
      </ul>
    </div>

    <div class="chat-area">
      <div v-if="isAskingTenderQuestion" class="mode-indicator">
        <el-alert title="当前处于招标文件提问模式" type="info" />
      </div>
      <div v-else class="mode-indicator">
        <el-alert title="当前处于知识库回答模式" type="info" />
      </div>
      <div class="messages">
        <div
          v-for="message in messages"
          :key="message.id"
          :class="['message', message.sender === 'user' ? 'user-message' : 'ai-message']"
        >
          <div class="message-header">
            <el-avatar class="mr-3" :src="message.avatar" :size="32" />
            <span class="sender-name">{{ message.sender === 'user' ? '用户' : 'AI' }}</span>
          </div>
          <div class="message-content">{{ message.text }}</div>
          <div v-if="message.sender === 'user' && isLoading" class="loading">
            加载中...
          </div>
        </div>
      </div>

      <div class="input-area">
        <el-input type="text" v-model="newMessage" @keyup.enter="sendMessage" placeholder="输入消息" />
        <el-button @click="sendMessage">发送</el-button>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, reactive } from 'vue';
import requestdb from '@/utils/requestdb.ts';
import requestai from '@/utils/requestai.ts';
import request from '@/utils/request.ts';
import { ElMessage, ElAvatar, ElButton, ElUpload } from 'element-plus';
import 'element-plus/es/components/message/style/css';

interface KnowledgeItem {
  id: number;
  name: string;
}

interface Message {
  id: number;
  text: string;
  sender: string;
  avatar: string;
}

interface ChatHistory {
  id: number;
  summary: string;
}

const knowledgeItems = ref<KnowledgeItem[]>([
  { id: 1, name: '暂无可用知识库' },
]);

const selectedItems = ref<number[]>([]);
const messages = ref<Message[]>([]);
const newMessage = ref('');
const chatHistory = ref<ChatHistory[]>([]);
const isLoading = ref(false);
const isAskingTenderQuestion = ref(false); // 新增状态变量

const AI_AVATAR = 'https://path.to/ai-avatar.png';

// 控制上传窗口的显示与隐藏
const uploadDialogVisible = ref(false);

// 获取用户头像
const fetchUserAvatar = async () => {
  try {
    const response = await request.post('/personal_page', { user: 'admin' });
    if (response.data.success === 'true') {
      return response.data.url;
    } else {
      return '';
    }
  } catch (error: any) {
    return '';
  }
};

// 删除文件函数
const deleteFile = async () => {
  try {
    const response = await request.post('http://127.0.0.1:5000/excel_match/delete_excel');
    if (response.data.success) {
      ElMessage.success('文件删除成功');
    } else {
      ElMessage.error('文件删除失败');
    }
  } catch (error) {
    ElMessage.error('请求出错: ' + error.message);
  }
};

const toggleSelectItem = (id: number) => {
  const index = selectedItems.value.indexOf(id);
  if (index === -1) {
    selectedItems.value.push(id);
  } else {
    selectedItems.value.splice(index, 1);
  }
};

const toggleSelectchat = async (id: any) => {
  const response = await requestdb.post('/show_his', { username: 'admin', session_id: id.toString() });
  messages.value = response.data.data;
};

const toggleTenderQuestionMode = () => {
  isAskingTenderQuestion.value = !isAskingTenderQuestion.value;
};

const sendMessage = async () => {
  if (newMessage.value.trim() === '') {
    return;
  }

  const userAvatar = await fetchUserAvatar();
  messages.value.push({ id: Date.now(), text: newMessage.value, sender: 'user', avatar: userAvatar });
  isLoading.value = true;

  const formData = new FormData();
  formData.append('username', 'admin');
  formData.append('number', '1');
  formData.append('cont', newMessage.value);
  formData.append('question', newMessage.value);
  
  for (const id of selectedItems.value) {
    formData.append('id', id.toString());
  }

  let apiUrl; // 定义 API URL
  if (isAskingTenderQuestion.value) {
    // 招标文件提问
    apiUrl = 'http://127.0.0.1:5000/excel_match/match_query';
  } else {
    // 常规提问
    apiUrl = 'http://127.0.0.1:5000/multimodels/index_milvus';
  }

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      body: formData,
    });

    const aiMessage = reactive({ id: Date.now(), text: '', sender: 'AI', avatar: AI_AVATAR });
    messages.value.push(aiMessage);

    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        aiMessage.text += decoder.decode(value);
      }
    }
  } catch (error) {
    ElMessage.error('请求错误: ' + error);
  } finally {
    newMessage.value = '';
    isLoading.value = false;
  }
};

const handleAddTenderDoc = () => {
  uploadDialogVisible.value = true;
};

// 文件上传成功处理
const handleUploadSuccess = (response: any, file: any) => {
  uploadDialogVisible.value = false; // 上传成功后关闭弹窗 
  ElMessage.success('文件上传成功');
};

// 文件上传失败处理
const handleUploadError = (error: any) => {
  uploadDialogVisible.value = false; // 上传失败后关闭弹窗 
  ElMessage.error('文件上传失败: ' + error);
};

const fetchdocsdata = async () => {
  try {
    const response = await requestdb.post('/show_filesss', { user: 'admin' });
    if (response.data.success === 'true' && response.data.data.length > 0) {
      knowledgeItems.value = response.data.data;
    }
  } catch (error: any) {
    console.log('发生错误: ' + error.message);
  }
};

const fetchChatHistory = async () => {
  try {
    const response = await request.post('/multimodels/show_session', { username: 'admin' });
    if (response.data.success === 'true') {
      chatHistory.value = response.data.data;
    }
  } catch (error: any) {
    console.log('发生错误: ' + error.message);
  }
};

onMounted(() => {
  fetchdocsdata();
  fetchUserAvatar();
  fetchChatHistory();
});
</script>

<style scoped>
.chat-app {
  display: flex;
  height: 79vh;
  background-color: #f5f5f5;
  font-family: Arial, sans-serif;
}

.sidebar, .history {
  width: 250px;
  border-right: 1px solid #e0e0e0;
  background-color: #ffffff;
  padding: 20px;
}

.sidebar h3, .history h3 {
  font-size: 18px;
  color: #333333;
  margin-bottom: 20px;
}

.sidebar ul, .history ul {
  list-style-type: none;
  padding: 0;
}

.sidebar li, .history li {
  padding: 10px;
  cursor: pointer;
  background-color: #ffffff;
  margin-bottom: 10px;
  border-radius: 5px;
  transition: background-color 0.3s ease;
}

.sidebar li.selected, .history li:hover {
  background-color: #d3e3fd;
}

.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
}

.messages {
  flex: 1;
  overflow-y: auto;
  background-color: #f5f5f5;
  padding: 10px;
  border-radius: 10px;
  border: 1px solid #e0e0e0;
}

.message {
  margin-bottom: 20px;
}

.message-header {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.message-content {
  padding: 10px;
  border-radius: 10px;
  background-color: #ffffff;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
}

.user-message .message-content {
  background-color: #d3e3fd;
}

.ai-message .message-content {
  background-color: #f0f0f0;
}

.input-area {
  display: flex;
  margin-top: 20px;
}

.input-area input {
  flex: 1;
  margin-right: 10px;
}

.mode-indicator {
  margin-bottom: 10px;
}

.delete-file-btn {
  margin-top: 10px;
}
</style>
