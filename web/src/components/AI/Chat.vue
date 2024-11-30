<template>
  <div class="chat-container">
    <!-- 添加条件渲染，当没有消息时显示图片 -->
    <div v-if="messages.length === 0" class="image-container">
      <img class="img" src="@/assets/AIlogo.svg" alt="Placeholder Image">
      <h1 style="font-size: 20px;">我今天能帮你什么？</h1>
    </div>
    <div class="message-container">
      <!-- 这里放聊天消息的显示区域，可以根据需要自行设计 -->
      <div v-for="(message, index) in messages" :key="index" class="message-wrapper">
        <div class="message-header">
          <el-avatar
            class="mr-3"
            :size="32"
            
          />
          <span class="sender-name">{{ message.sender === 'user' ? '用户' : 'AI' }}</span>
        </div>
        <div :class="['message-bubble', message.sender]">
          <div class="message-content">
            <div v-if="message.type === 'text'">{{ message.content }}</div>
            <img v-if="message.type === 'image'" :src="message.content" alt="Generated Image" />
          </div>
        </div>
      </div>
    </div>
    <div class="button-container">
     <img src="@/assets/聊天历史.svg" @click="showChatHistory" />
     <img src="@/assets/新聊天.svg" @click="startNewChat" />
     <img src="@/assets/阅读.svg"  @click="readDocument" />
     <img src="@/assets/已上传文件.svg"  @click="triggerFileUpload" />
     <img src="@/assets/文生图.svg" @click="generateImage" />
    </div>
    <div v-if="file" class="file-info">{{ file.name }}</div>
    <div class="input-container">
      <!-- 隐藏的文件输入框 -->
      <input ref="fileInput" type="file" @change="handleFileUpload" style="display: none;" />
      <el-input 
        v-model="input" 
        type="textarea" 
        :autosize="{ minRows: 5, maxRows: 10 }" 
        placeholder="输入聊天..." 
        @keypress="handleKeyPress" 
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import requestai from '@/utils/requestai.ts'
import { ElMessage } from 'element-plus';
const input = ref('');
const num = ref("1");
const messages = ref<{ sender: string, content: string, type: string }[]>([]);


const file = ref<File | null>(null);
const fileInput = ref<HTMLInputElement | null>(null); // 确保 fileInput 被正确引用


const sendMessage = () => {
  if (input.value.trim() === '') return;

  // 本地显示发送的消息
  messages.value.push({ sender: 'user', content: input.value, type: 'text' });

  let formData = new FormData();
  formData.append("username", "admin");
  formData.append("number", num.value);
  formData.append("cont", input.value);

  if (file.value) {
    formData.append("file", file.value);
    requestai({
    method: 'post',
    url: "/getAI",
    data: formData,
  }).then(res => {
    console.log(res.data);
    messages.value.push({ sender: 'ai', content: res.data.answer, type: 'text' });
    console.log(res.data.answer);
  }).catch(error => {
    ElMessage.error("请求错误: ", error);
  });
  input.value = '';
  file.value = null;
  }

else{
  requestai({
    method: 'post',
    url: "/getAI",
    data: formData,
  }).then(res => {
    console.log(res.data);
    messages.value.push({ sender: 'ai', content: res.data.answer, type: 'text' });
    console.log(res.data.answer);
  }).catch(error => {
    ElMessage.error("请求错误: ", error);
  });
  input.value = '';
  file.value = null;
}
  
};

const handleKeyPress = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault(); // 阻止默认的换行行为
    sendMessage();
  }
};

const triggerFileUpload = () => {
  if (fileInput.value) {
    fileInput.value.click();
  }
};
const emit = defineEmits(['show-pdf']);
const handleFileUpload = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    file.value = target.files[0];
  }
};

const showChatHistory = () => {
  // 显示聊天历史逻辑
  console.log("显示聊天历史");
};

const startNewChat = () => {
  // 新建聊天逻辑
  console.log("新建聊天");
  messages.value = [];
  file.value = null;
};

const readDocument = () => {
  // 阅读此文档逻辑
  console.log("阅读此文档");
  const blob = new Blob([document.documentElement.outerHTML], { type: 'text/html' });
  file.value = new File([blob], "document.html", { type: 'text/html' });
};

const generateImage = () => {
  if (input.value.trim() === '') return;

  // 本地显示发送的消息
  messages.value.push({ sender: 'user', content: input.value, type: 'text' });

 requestai({
    method: 'post',
    url: "/wordtopic",
    data: { prompt: input.value },
  }).then(res => {
    // 假设后端返回的图片URL是完整的
    const imageUrl = res.data;
    
    // 如果后端返回的图片URL不完整，这里可以构造完整的URL
    // const baseUrl = 'http://example.com'; // 后端提供的图片URL的基础部分
    // const fullImageUrl = `${baseUrl}/${imageUrl}`;

    messages.value.push({ sender: 'ai', content: imageUrl, type: 'image' });
    console.log(res.data);
  }).catch(error => {
    console.error("请求错误: ", error);
  });

  // 发送完成后清空输入框内容
  input.value = '';
};

</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 90vh; /* 让聊天界面占满整个页面高度 */
  width: 100%; /* 让聊天界面占满整个页面宽度 */
}

.button-container {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  background-color: #ffffff;
}

.img {
  width: 40%;
}

.image-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1; /* 占据剩余的空间 */
}

.button-container img {
  width: 10%;
}

.message-container {
  flex: 1; /* 消息区域占据剩余的空间 */
  overflow-y: auto; /* 如果消息过多，自动显示滚动条 */
  padding: 10px;
  background-color: #f5f5f5;
  
}

.input-container {
  display: flex;
  align-items: center;
  padding: 10px;
  background-color: #ffffff;
  margin-bottom: 20%;
  margin-right: 10%;
}

.message-wrapper {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-bottom: 10px;
}

.message-header {
  display: flex;
  align-items: center;
  margin-bottom: 5px;
}

.message-bubble {
  max-width: 70%;
  padding: 10px;
  border-radius: 10px;
  word-wrap: break-word;
  position: relative;
}

.message-bubble.user {
  background-color: #9fe7a4;
  color: #000;
}

.message-bubble.ai {
  background-color: #e0e0e0;
  color: #000;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  margin-right: 8px;
}

.sender-name {
  font-weight: bold;
}

.message-content {
  white-space: pre-wrap; /* 保留空白字符 */
}

.file-info {
  margin-top: 10px;
  font-size: 14px;
  color: #888;
}
</style>
