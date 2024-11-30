<template>
  <div>
    <h1>从图像中提取文本，数字和方程</h1>

    <el-upload
      class="upload-demo"
      drag
      action=""
      multiple
      :http-request="uploadRequest"
      style="width: 240px"
    >
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">
        将图片拖拽至此处或<em>点击上传</em>
      </div>
      <template #tip>
        <div class="el-upload__tip">
          最大尺寸500kb、支持jpg、png格式
        </div>
      </template>
    </el-upload>
    结果：{{ aiResponse }}
  </div>
</template>

<script setup lang="ts">
import { UploadFilled } from '@element-plus/icons-vue'
import { ref } from 'vue'

import   requestai  from '@/utils/requestai.ts'
const aiResponse = ref('');


const uploadRequest = (options: any) => {
  const file = options.file;
  let formData = new FormData();
  formData.append("file", file);
  formData.append("number", "3");
  formData.append("username", "admin");
  formData.append("cont", "");

  requestai({
    method: 'post',
    url: "/getAI",
    data: formData,
  }).then(res => {
    aiResponse.value = res.data.answer;
    options.onSuccess(res.data, file);
    console.log('后端返回的消息：', res.data.answer); // 打印后端返回的消息
  }).catch(error => {
    console.error("请求错误: ", error);
    options.onError(error);
  });
};

</script>
