<template>
  <div style="width: 240px">
    <el-input v-model="input"   type="textarea"  :autosize="{ minRows: 4, maxRows:8 }"  placeholder="输入以聊天" style="margin-bottom: 20px;" />

    <h2>格式</h2>
    <div class="write">
      <el-button :class="{ selected: format === 'outline' }" @click="selectFormat('outline')">大纲</el-button>
      <el-button :class="{ selected: format === 'continue' }" @click="selectFormat('continue')">续写</el-button>
      <el-button :class="{ selected: format === 'polish' }" @click="selectFormat('polish')">润色</el-button>
      <el-button :class="{ selected: format === 'email' }" @click="selectFormat('email')">邮件</el-button>
      <el-button :class="{ selected: format === 'outline' }" @click="selectFormat('abstrct')">摘要</el-button>
    </div>

    <h2>语气</h2>
    <div class="write">
      <el-button :class="{ selected: tone === 'formal' }" @click="selectTone('formal')">正式的</el-button>
      <el-button :class="{ selected: tone === 'casual' }" @click="selectTone('casual')">随意的</el-button>
      <el-button :class="{ selected: tone === 'professional' }" @click="selectTone('professional')">专业的</el-button>
    </div>

    <h2>长度</h2>
    <div class="write">
      <el-button :class="{ selected: length === 'short' }" @click="selectLength('short')">短</el-button>
      <el-button :class="{ selected: length === 'medium' }" @click="selectLength('medium')">中</el-button>
      <el-button :class="{ selected: length === 'long' }" @click="selectLength('long')">长</el-button>
    </div>
    <h2>语言</h2>
    <div class="write">
      <el-button :class="{ selected: language === 'chinese' }" @click="selectLanguage('chinese')">中文</el-button>
      <el-button :class="{ selected: language === 'english' }" @click="selectLanguage('english')">英文</el-button>
      <el-button :class="{ selected:language === 'japanese' }" @click="selectLanguage('japanese')">日文</el-button>
    </div>

    <el-button type="primary" @click="sendMessage">发送</el-button>

    <div class="response">
      结果：{{ aiResponse }}
    </div>
  </div>
</template>

<script setup lang="ts">

import { ref } from 'vue';
import  requestai  from '@/utils/requestai.ts';
const input = ref('');
const num = ref("4");
const aiResponse = ref('');
const format = ref('');
const tone = ref('');
const length = ref('');
const language=ref('');
const sendMessage = () => {
  if (input.value.trim() === '') return;

  let formData = new FormData();
  formData.append("username", "123456");
  formData.append("number", num.value);
  formData.append("cont", input.value);
  formData.append("format", format.value);
  formData.append("tone", tone.value);
  formData.append("length", length.value);
  formData.append("language", language.value);
  requestai({
    method: 'post',
    url: "/getAI",
    data: formData,
  }).then(res => {
    aiResponse.value = res.data.answer;
    console.log(res.data.answer);
  }).catch(error => {
    console.error("请求错误: ", error);
  });

  // 发送完成后清空输入框内容
  input.value = '';
};

const selectFormat = (selectedFormat: string) => {
  format.value = selectedFormat;
};

const selectTone = (selectedTone: string) => {
  tone.value = selectedTone;
};

const selectLength = (selectedLength: string) => {
  length.value = selectedLength;
};
const selectLanguage = (selectedLanguage: string) => {
  language.value = selectedLanguage;
};
</script>

<style scoped>
.write {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
}

.el-button.selected {
  background-color: #409EFF;
  color: #ffffff;
}

.response {
  margin-top: 20px;
}
</style>
