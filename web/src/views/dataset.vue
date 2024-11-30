<template>
  <el-card style="width: auto">
    <template #header>
      <div style="display: flex;">
        <el-upload
          action="/create_words"
          :on-success="handleUploadSuccess"
          :on-error="handleUploadError"
          :data="uploadData"
          :show-file-list="false"
          :http-request="customRequest"
        >
          <el-button type="primary">上传</el-button>
        </el-upload>
        <el-button type="danger" @click="deleteKnowledgeBase" style="margin-left: 10px;">删除知识库</el-button>
        <el-button
          :type="removeWatermark === 1 ? 'success' : 'danger'"
          @click="toggleWatermark"
          style="margin-left: 10px;"
        >
          {{ removeWatermark === 1 ? '去除水印：是' : '去除水印：否' }}
        </el-button>
      </div>
    </template>

    <el-table v-show="!loading" :data="alldocs" border style="width: auto; display: flex;">
      <el-table-column prop="date" label="修改日期" width="180" align="center" />
      <el-table-column prop="name" label="名称" width="400" align="center" />
      <el-table-column prop="sorts" label="类型" align="center" />
      <el-table-column prop="address" label="操作" align="center">
        <template #default="{ row }">
          <el-button @click="previewChunkEffect(row.name)">预览分块效果</el-button>
          <el-button @click="deleteDocument(row.name)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-skeleton :loading="loading" :rows="5" animated />
  </el-card>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useRoute } from 'vue-router';
import requestdb from '@/utils/requestdb';
import { ElMessage } from 'element-plus';
import router from '@/router';

const loading = ref(true);

// 全局变量，用来控制去除水印状态
const removeWatermark = ref(0); // 0 表示否，1 表示是

interface Document {
  date: string;
  name: any;
  sorts: string;
}

const route = useRoute();
const alldocs = ref<Array<Document>>([]);
const id = ref(route.params.id);

const fetchdocsdata = async () => {
  try {
    const response = await requestdb.post('/show_milvus', { user: 'admin', id: id.value });
    if (response.data.success === 'true' && response.data.data.length > 0) {
      alldocs.value = response.data.data;
    } else {
      console.log(`获取数据失败: ${response.data.message}`);
    }
  } catch (error) {
    const err = error as Error;
    console.log('发生错误: ' + err.message);
  } finally {
    setTimeout(() => {
      loading.value = false;
    }, 1000);  // 保证骨架屏至少显示1秒
  }
};

const uploadData = (file: File) => {
  return {
    username: 'admin',
    ids: id.value,
    times: new Date().toISOString(),
    various: getFileType(file.name),
    dconame: file.name,
  };
};

const customRequest = async (option: any) => {
  ElMessage.success('文件已成功上传，正在嵌入，请稍候...');
  const formData = new FormData();
  formData.append('username', 'admin');
  formData.append('ids', id.value.toString());
  formData.append('times', new Date().toISOString());
  formData.append('various', getFileType(option.file.name)); 
  formData.append('dconame', option.file.name); 
  formData.append('doc', option.file);
  formData.append('aaa',removeWatermark.value);
  try {
    const response = await requestdb.post(option.action, formData);
    if (response.data.success === 'true') {
      ElMessage.success('文件嵌入成功！');
      // 刷新页面
      fetchdocsdata();
    } else {
      ElMessage.success('文件嵌入成功！');
      // 刷新页面
      fetchdocsdata();
    }
  } catch (error: any) {
    option.onError(error);
    ElMessage.error(`出现错误：${error.message}`);
  }
};

const previewChunkEffect = (name: string) => {
  ElMessage.info(`预览分块效果：${name}`);
  router.push({ name: 'Preview', params: { id: id.value, dconame: name } });
};

const deleteDocument = async (name: string) => {
  const formData = new FormData();
  formData.append('username', 'admin');
  formData.append('ids', id.value.toString());
  formData.append('dconame', name); 
  
  try {
    const response = await requestdb.post('/delete_file', formData);
    if (response.data.success) {
      ElMessage.success('删除成功');
      // 从 alldocs 中移除已删除的文件
      alldocs.value = alldocs.value.filter(doc => doc.name !== name);
    } else {
      ElMessage.error(`删除失败: ${response.data.message}`);
    }
  } catch (error) {
    const err = error as Error;
    ElMessage.error('删除失败: ' + err.message);
  }
};

const deleteKnowledgeBase = async () => {
  const formData = new FormData();
  formData.append('username', 'admin');
  formData.append('ids', id.value.toString());
  
  try {
    const response = await requestdb.post('/delete_word', formData);
    if (response.data.success) {
      ElMessage.success('知识库删除成功');
      window.location.href = 'http://localhost:5173/editFront1/knowledge';
    } else {
      ElMessage.error(`删除失败: ${response.data.message}`);
    }
  } catch (error) {
    const err = error as Error;
    ElMessage.error('删除失败: ' + err.message);
  }
};

// 切换去除水印按钮的状态
const toggleWatermark = () => {
  removeWatermark.value = removeWatermark.value === 1 ? 0 : 1;
  ElMessage.success(`去除水印已设置为：${removeWatermark.value === 1 ? '是' : '否'}`);
};

onMounted(() => {
  fetchdocsdata();
});

const getFileType = (fileName: string) => {
  const extension = fileName.split('.').pop()?.toLowerCase();
  switch (extension) {
    case 'doc':
    case 'docx':
      return 'Word 文档';
    case 'ppt':
    case 'pptx':
      return 'PPT 文件';
    case 'xls':
    case 'xlsx':
      return 'Excel 文件';
    case 'pdf':
      return 'PDF 文件';
    case 'jpg':
      return 'jpg 文件';
    default:
      return '未知文件类型';
  }
};

const handleUploadSuccess = (response: any, file: any) => {
  if (response.success) {
    ElMessage.success('文件上传成功，正在处理文件...');
    const fileType = getFileType(file.name);
    alldocs.value.push({
      date: new Date().toISOString().split('T')[0],
      name: file.name,
      sorts: fileType,
    });
    loading.value = true;
    setTimeout(() => {
      loading.value = false;
      ElMessage.success('文件处理完成');
      fetchdocsdata();
    }, 300000); // 模拟处理时间
  } else {
    ElMessage.error(`上传失败: ${response.message}`);
  }
};

const handleUploadError = (error: any) => {
  ElMessage.error(`上传失败: ${error.message}`);
};
</script>

<style lang="scss">
</style>
