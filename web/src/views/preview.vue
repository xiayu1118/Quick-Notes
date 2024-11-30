<template>
  <div id="app">
    <div class="document">
      <h2>文档内容</h2>
      <div v-if="pdfUrl" ref="pdfContainer"></div>
    </div>

    <div class="blocks">
      <h2>分块效果</h2>
      <div v-for="(block, index) in blocks" :key="index" class="block">
        <h3>块 {{ index + 1 }}</h3>
        <p>{{ block }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref, nextTick } from 'vue';
import { useRoute } from 'vue-router';
import requestdb from '@/utils/requestdb.ts';
import PDFObject from 'pdfobject';

const route = useRoute();
const blocks = ref<string[]>([]);
const pdfUrl = ref<string | null>(null);
const pdfContainer = ref(null);

const fetchdata = async () => {
  const formData = new FormData();
  formData.append("username", "admin");
  formData.append("id", route.params.id as string);
  formData.append("dconame", route.params.dconame as string);

  try {
    const response = await requestdb.post('/get_chunks', formData);
    blocks.value = response.data.chunk;
    pdfUrl.value = response.data.file_url;

    if (pdfUrl.value) {
      await nextTick();
      PDFObject.embed(pdfUrl.value, pdfContainer.value);
    }
  } catch (error) {
    const err = error as Error;
    console.log('发生错误: ' + err.message);
  }
};

onMounted(() => {
  fetchdata();
});
</script>

<style>
#app {
  display: flex; /* 设置为flex布局 */
  justify-content: space-between; /* 将子元素分布在容器两端 */
  align-items: flex-start; /* 确保子元素在交叉轴的开始位置对齐 */
  width: 100%; /* 确保容器占满整个父元素宽度 */
}

.document, .blocks {
  float:left;border:1px solid red; padding:10px;
  width: 50%;
  height:100%
}
.pdfobject-container {
   height: 800px;
    border: 1px solid #ccc;
   }
h2 {
  font-size: 1.5em;
  margin-bottom: 15px;
  color: #555;
}

h3 {
  font-size: 1.2em;
  margin-bottom: 10px;
  color: #666;
}

p {
  line-height: 1.6;
  margin-bottom: 10px;
}

.block {
  border: 1px solid #ccc;
  padding: 15px;
  margin-bottom: 15px;
  border-radius: 5px;
  background-color: #f9f9f9;
}

.block:hover {
  background-color: #e9e9e9;
}

.block h3 {
  margin-top: 0;
}
</style>
