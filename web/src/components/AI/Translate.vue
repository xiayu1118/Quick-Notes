<template>
    <h1>翻译</h1>
    <el-select
      v-model="value1"
      placeholder="Select"
      size="large"
      style="width: 240px"
    >
      <el-option
        v-for="item in options1"
        :key="item.value"
        :label="item.label"
        :value="item.value"
      />
    </el-select>
    <el-select
      v-model="value2"
      placeholder="Select"
      size="large"
      style="width: 240px"
    >
      <el-option
        v-for="item in options2"
        :key="item.value"
        :label="item.label"
        :value="item.value"
      />
    </el-select>
    <el-input v-model="input" type="textarea" :autosize="{ minRows: 6, maxRows: 8 }" placeholder="输入你要翻译的语句" />   
    <el-icon ><SetUp /></el-icon>
    <el-button type="primary" @click="sendMessage">提交</el-button>
    <h1>结果：{{ aiResponse }}</h1>
         
</template>
<script setup lang="ts">
import { ref } from 'vue'
import  requestai  from '@/utils/requestai.ts'
const num = ref("2")
const value1 = ref('')
const value2 = ref('');
const input = ref('');
const aiResponse = ref('');
const options1 = [
  {
    value: 'Chinese',
    label: '中文(简体)',
  },
  {
    value: 'Chinese2',
    label: '中文(繁体)',
  },
  {
    value: 'English',
    label: 'English',
  },
  {
    value: 'japanese',
    label: '日本語にほんご',
  },
  {
    value: 'Spanish',
    label: 'español',
  }
]
const options2 = [
  {
    value: 'Chinese',
    label: '中文(简体)',
  },
  {
    value: 'Chinese2',
    label: '中文(繁体)',
  },
  {
    value: 'English',
    label: 'English',
  },
  {
    value: 'japanese',
    label: '日本語にほんご',
  },
  {
    value: 'Spanish',
    label: 'español',
  },
]

const sendMessage = () => {
    // 这里添加与后端交互的代码，示例中假设使用 fetch 或 axios 发送 POST 请求
    let formData = new FormData();
  formData.append("username", "123456");
  formData.append("number", num.value);
  formData.append("language1", value1.value);
  formData.append("language2", value2.value);
  formData.append("cont", input.value);
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
</script>