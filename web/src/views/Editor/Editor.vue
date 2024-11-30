<template>
  <div class="relative">
    <div class="editor-container mx-4">
      <Toolbar style="border-bottom: 1px solid #ccc" :editor="editorRef" :defaultConfig="toolbarConfig" />
      <el-input v-model="documentName" placeholder="请输入文档名称" style="margin-bottom: 10px;" />
      <el-button type="submit" @click="saveDocument">保存</el-button>

        <div id="editor-container">
          <div id="title-container"></div>
          <Editor
            style="height: 900px; justify-content: space-between; overflow-y: hidden;"
            v-model="valueHtml"
            :defaultConfig="editorConfig"
            @onCreated="handleCreated"
          />
        </div>
      </div>
    </div>
</template>

<script setup>
import { ref, watch, computed, onMounted, onBeforeUnmount, shallowRef } from 'vue';
import '@wangeditor/editor/dist/css/style.css';
import { Editor, Toolbar } from '@wangeditor/editor-for-vue';
import axios from 'axios';
import { useRoute } from 'vue-router';
import { useTokenStore } from '@/store/userstoken';
import request from '@/utils/request';
import { ElMessage } from 'element-plus';
const input = ref('');
const aiResponse = ref('');
const isWindowVisible = ref(false);
const documentName = ref('');
const editorRef = shallowRef();
const valueHtml = ref('<p></p>');
const store = useTokenStore();
const route = useRoute();
documentName.value = route.query.documentName || '';

onMounted(() => {
  if (documentName.value) {
    request.post('/get_document', { name: documentName.value, user: store.token.access_token })
      .then((response) => {
        if (response.data.content) {
          valueHtml.value = response.data.content;
          console.log(valueHtml.value)
        }
      })
      .catch((error) => {
        console.error('Error fetching document:', error);
      });
  }

 
});

const toolbarConfig = {};
const editorConfig = { placeholder: '在此输入' };

onBeforeUnmount(() => {
  const editor = editorRef.value;
  if (editor == null) return;
  editor.destroy();
});

const handleCreated = (editor) => {
  editorRef.value = editor;
};
const saveDocument = () => {
  if (!documentName.value) {
    alert("请填写文档名称");
    return;
  }
  request.post('/upload_file', {
    name: documentName.value,
    content: valueHtml.value,
    user: store.token.access_token,
    sorts: 'word'
  })
  .then((response) => {
    ElMessage.success('文档保存成功!');
    console.log(response.data);
  })
  .catch((error) => {
    if (error.response) {
      console.error('服务器返回错误:', error.response.data);
      ElMessage.error('保存文档出错: ' + error.response.data.message);
    } else if (error.request) {
      console.error('请求发送失败:', error.request);
    } else {
      console.error('请求错误:', error.message);
    }
  });
};
</script>

<style>
@import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.0/dist/tailwind.min.css');

h1, p {
  font-family: Lato;
}

.floating-window {
  z-index: 50;
}

#content {
  height: calc(100% - 40px);
  background-color: rgb(245, 245, 245);
  overflow-y: auto;
  position: relative;
}

#editor-container {
  width: 850px;
  margin: 30px auto 150px auto;
  background-color: #fff;
  padding: 20px 50px 50px 50px;
  border: 1px solid #e8e8e8;
  box-shadow: 0 2px 10px rgb(0 0 0 / 12%);
}

#title-container {
  padding: 20px 0;
  border-bottom: 1px solid #e8e8e8;
}
</style>
