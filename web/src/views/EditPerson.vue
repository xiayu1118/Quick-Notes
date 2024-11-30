<template>
  <div class="container">
    <div class="flex items-center">
      <el-avatar class="mr-3" :size="250" :src="avatarUrl" @click="openFileDialog" />
      <span class="edit-avatar-hint">点击头像以编辑</span>
      <input type="file" ref="fileInput" @change="handleFileChange" style="display: none;" />
      <div class="form-container">
        <form @submit.prevent="handleSubmit">
          <div class="form-group">
            <label for="account">账号</label>
            <input id="account" v-model="formData.account" type="text" readonly required />
          </div>
          <div class="form-group">
            <label for="password">密码</label>
            <input id="password" v-model="formData.password" type="password" readonly required />
          </div>
          <div class="form-group">
            <label for="email">邮箱</label>
            <input id="email" v-model="formData.email" type="email" :readonly="!isEditing" required />
          </div>
          <div class="form-group">
            <label for="phone">手机号</label>
            <input id="phone" v-model="formData.phone" type="tel" :readonly="!isEditing" required />
          </div>
          <div class="form-group">
            <label for="token">百度秘钥token</label>
            <input id="token" v-model="formData.token" type="text" :readonly="!isEditing" required />
          </div>
          <div class="form-group">
            <label for="nickname">昵称</label>
            <input id="nickname" v-model="formData.nickname" type="text" :readonly="!isEditing" required />
          </div>
          <button v-if="!isEditing" type="button" @click="startEdit">编辑</button>
          <button v-if="isEditing" type="button" @click="cancelEdit">取消修改</button>
          <button v-if="isEditing" type="submit">保存修改</button>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';

import { useTokenStore, useUserAvatarStore } from '@/store/userstoken';
import request from '@/utils/request.ts';
import {ElMessage} from "element-plus"
const defaultValues = {
  account: '默认账号',
  password: '默认密码',
  email: 'example@example.com',
  phone: '1234567890',
  token: '默认Token',
  pet_name: '默认展示昵称',
  avatar: 'https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png'
};

const formData = ref({
  account: defaultValues.account,
  password: defaultValues.password,
  email: defaultValues.email,
  phone: defaultValues.phone,
  token: defaultValues.token,
  nickname: defaultValues.pet_name,

});

const avatarUrl = ref(defaultValues.avatar);
const isEditing = ref(false); // 初始设为不可编辑
const store = useTokenStore();
const avatarStore = useUserAvatarStore();
const fileInput = ref<HTMLInputElement | null>(null);
const fetchPersonalPageData = async () => {
  try {
    console.log("访问个人信息展示界面，更新数据");
    const response = await request.post('/personal_page', { user: store.token.access_token });
    if (response.data.success === 'true' && response.data.data.length > 0) {
      const data = response.data.data[0];
      console.log("后端响应数据：",data);
      formData.value.account = data.account ?? defaultValues.account;
      formData.value.password = data.password ?? defaultValues.password;
      formData.value.email = data.email ?? defaultValues.email;
      formData.value.phone = data.phone ?? defaultValues.phone;
      formData.value.token = data.token ?? defaultValues.token;
      formData.value.nickname = data.pet_name ?? defaultValues.pet_name;
      avatarUrl.value = data.url;
      console.log('avatarUrl:', avatarUrl.value); // 输出 URL 以便调试
    } else {
      ElMessage.error(`获取数据失败: ${response.data.message}`);
    }
  } catch (error) {
    const err=error as Error;
    ElMessage.error('发生错误: ' + err.message);
  }
};

const updatePersonalPageData = async () => {
  try {
    console.log("更新用户信息");
    const updatedData = {
      account: formData.value.account,
      email: formData.value.email,
      phone: formData.value.phone,
      token: formData.value.token,
      pet_name: formData.value.nickname
    };
    const response = await request.post('/person_page_show', updatedData);
    if (response.data.success === 'true') {
      ElMessage.success('用户信息更新成功');
      isEditing.value = false;
      await fetchPersonalPageData();
    } else {
      ElMessage.error(`更新数据失败: ${response.data.message}`);
    }
  } catch (error) {
    const err=error as Error;
    ElMessage.error('发生错误: ' + err.message);
  }
};

const handleFileChange = async () => {
  
  const file = fileInput.value?.files?.[0];
  const formData = new FormData();
  if (file && file instanceof File) {
  formData.append('avatar', file);
} else {
  // 处理file为undefined或不是File类型的情况
}
  formData.append('user', store.token.access_token);

  try {
    const response = await request.post('/upload_avatar', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    if (response.data.success === 'true') {
      ElMessage.success('头像上传成功');
      avatarUrl.value = response.data.avatar_url;
      avatarStore.setAvatarUrl(response.data.avatar_url);
      console.log('头像 URL 更新为：', response.data.avatar_url);
    } else {
      console.error('头像上传失败:', response.data);
      ElMessage.error(`头像上传失败: ${response.data.message}`);
    }
  } catch (error) {
    const err = error as Error;
  
    ElMessage.error('发生错误: ' + err.message);
  }
};

const openFileDialog = () => {
  fileInput.value?.click();
};

const startEdit = () => {
  isEditing.value = true;
};

const cancelEdit = () => {
  isEditing.value = false;
  fetchPersonalPageData(); // 取消编辑时重新加载数据
};

const handleSubmit = () => {
  updatePersonalPageData();
};

onMounted(() => {
  fetchPersonalPageData();
});
</script>

<style scoped>
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  height: 78vh;
  text-align: center;
  background-color: #f9f9f9;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
  border-radius: 15px;
  font-family: 'Arial', sans-serif;
  border: 1px solid #ccc;
  background: url('@/assets/bg2.jpg') no-repeat center center/cover;
}

.flex {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.edit-avatar-hint {
  margin-left: -200px;
  margin-right: 200px;
  margin-top: -60px;
}

.form-container {
  flex: 1;
}

.form-group {
  margin-bottom: 20px;
  text-align: left;
}

label {
  display: block;
  font-size: 1em;
  margin-bottom: 8px;
  color: #333;
}

input {
  width: 90%;
  padding: 12px;
  font-size: 1em;
  border: 1px solid #ccc;
  border-radius: 5px;
  transition: border-color 0.3s;
}

input:focus {
  border-color: #007bff;
}

button {
  padding: 12px 24px;
  font-size: 1.2em;
  color: #fff;
  background-color: #007bff;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  margin-top: 20px;
  transition: background-color 0.3s;
}

button:hover {
  background-color: #0056b3;
}

button:first-of-type {
  background-color: #28a745;
}

button:first-of-type:hover {
  background-color: #218838;
}

.el-avatar {
  cursor: pointer;
  margin-top: -400px;
}
</style>
