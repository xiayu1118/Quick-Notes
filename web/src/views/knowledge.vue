<template>
  <div class="container">
    <!-- 搜索框和新建模块按钮 -->
    <div class="top-section">
      <el-input
        placeholder="搜索知识库"
        v-model="searchQuery"
        clearable
        class="search-box"
      ></el-input>
    </div>

    <div class="icon-grid-container">
      <!-- 上半部分：文档格式图标 -->
      <div class="icon-section">
        <h3>新建</h3>
        <div class="icon-list">
          <!-- 新建模块按钮 -->
          <div class="custom-card new-card" @click="dialogVisible = true">
            <i class="el-icon-plus">
              <el-icon><Plus /></el-icon>
            </i>
          </div>
          <el-card class="custom-card" v-for="item in filteredIcons" :key="item.id" @click="goToDocs(item.id)">
            <template #header>
              <div class="card-header">{{ item.name }}</div>
            </template>
            <img :src="item.imgSrc" style="width: 100%" />
          </el-card>
        </div>
      </div>
    </div>

    <!-- 弹出输入框 -->
    <el-dialog title="新建知识库" v-model="dialogVisible">
      <el-input v-model="newRepoName" placeholder="输入知识库名字"></el-input>
      <div slot="footer" class="dialog-footer">
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="createNewRepo">确定</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { ElCard, ElInput, ElDialog, ElButton, ElMessage } from 'element-plus';
import 'element-plus/dist/index.css';
import { useRouter } from 'vue-router';

import { useTokenStore } from '@/store/userstoken';
import requestdb from '@/utils/requestdb';
const store = useTokenStore();
// 模拟数据
const icons = ref([
  { id: 1, name: '示例知识库', imgSrc: 'https://shadow.elemecdn.com/app/element/hamburger.9cf7b091-55e9-11e9-a976-7f4d0b07eef6.png' },
  // 添加更多图标数据
]);

const searchQuery = ref('');

const filteredIcons = computed(() => {
  return icons.value.filter(icon => icon.name.includes(searchQuery.value));
});

const router = useRouter();

const dialogVisible = ref(false);
const newRepoName = ref('');

const fetchdocsdata = async () => {
   try {
     const response = await requestdb.post('/show_filesss', { user: store.token.access_token });
     if (response.data.success === 'true' && response.data.data.length > 0) {
      icons.value = response.data.data;
     } else {
       console.log(`获取数据失败: ${response.data.message}`);
     }
   } catch (error) {
    const err = error as Error;
    console.log('发生错误: ' + err.message);
   }
 };
const createNewRepo = async () => {
  if (newRepoName.value) {
    const newId = icons.value.length>0 ? icons.value[icons.value.length - 1].id + 1 : 1;
    icons.value.push({ id: newId, name: newRepoName.value, imgSrc: 'https://shadow.elemecdn.com/app/element/hamburger.9cf7b091-55e9-11e9-a976-7f4d0b07eef6.png' });
    console.log(newId)
    const username = 'admin';  // 替换为获取实际输入的方法
    const ids = newId; // 假设newRepoName.value就是ids
    const kname=newRepoName.value;
    const formData = new FormData();
    formData.append('username', username);
    formData.append('ids', ids.toString());
    formData.append('kname', kname);
    try {
      const response = await requestdb.post('/create_milvus', formData);
      if (response.data.success) {
        ElMessage.success(response.data.message);
        console.log(response.data.path);
        fetchdocsdata();
      } else {
        ElMessage.success(response.data.message);
        fetchdocsdata();
      }
    } catch (error:any) {
      console.error('Request failed:', error);
      if (error.response) {
        ElMessage.error('Request failed: ' + error.response.data.message);
      } else {
        ElMessage.error('Network error');
      }
    }

    newRepoName.value = '';
    dialogVisible.value = false;
  }
};

const goToDocs = (id:any) => {
  router.push(`/dataset/${id}`);
};

onMounted(() => {
  fetchdocsdata();
});
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.top-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.search-box {
  width: 300px;
}

.icon-grid-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.icon-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

.icon-list {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 15px;
  justify-items: center;
}

.custom-card {
  max-width: 200px;
  width: 100%;
  height: 100%;
}

.new-card {
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid #dcdfe6; /* 加粗边框 */
  border-radius: 5px; /* 圆角边框 */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* 添加阴影 */
  cursor: pointer;
  transition: box-shadow 0.3s ease; /* 平滑过渡效果 */
}

.new-card:hover {
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); /* 鼠标悬停时增加阴影 */
}

.el-icon-plus {
  font-size: 24px; /* 调整图标大小 */
  color: #409eff; /* 调整图标颜色 */
}

.card-header {
  font-size: 0.675rem;
  text-align: center;
}

.dialog-footer {
  text-align: right;
}
</style>
