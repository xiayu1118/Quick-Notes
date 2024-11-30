<template>
  <div :class="personClass" aria-label="A complete example of page header">
    <div class="custom-page-header">
      <div class="header-extra">
        <div class="button-container">
          <el-button @click="$router.push({ path: '/EditPerson' })" class="ml-2">
            <el-icon><EditPen /></el-icon>编辑你的个人信息
          </el-button>
          <el-button class="ml-2">你的工作</el-button>
        </div>
      </div>

      <div :class="personClass">
        <div class="avatar-container">
          <el-avatar :class="avatarClass" :src="avatarUrl" />
          <div class="tags">
            <span class="text-large font-1000 mr-3">{{ nickname }}</span>
            <span class="text-sm mr-3" style="color: var(--el-text-color-regular)">{{ account }}</span>
          </div>
        </div>
        <div class="info-container">
          <el-descriptions :column="1" size="large" class="mt-4">
            <el-descriptions-item label="账号">{{ account }}</el-descriptions-item>
            <el-descriptions-item label="密码">{{ password }}</el-descriptions-item>
            <el-descriptions-item label="电话号码">{{ phone }}</el-descriptions-item>
            <el-descriptions-item label="所在地">中国 湖南</el-descriptions-item>
            <el-descriptions-item label="电子邮箱">{{ email }}</el-descriptions-item>
            <el-descriptions-item label="百度飞桨token">{{ token }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
    </div>
  </div>

  <el-tabs v-model="activeName" class="demo-tabs" @tab-click="handleClick">
    <el-tab-pane label="个人信息" name="first">User</el-tab-pane>
    <el-tab-pane label="Config" name="second">Config</el-tab-pane>
    <el-tab-pane label="Role" name="third">Role</el-tab-pane>
    <el-tab-pane label="Task" name="fourth">Task</el-tab-pane>
  </el-tabs>
</template>

<script setup lang="ts">
import { ElMessage } from 'element-plus';
import { onMounted, ref } from 'vue';
import type { TabsPaneContext } from 'element-plus';
import { useTokenStore, useUserAvatarStore } from '@/store/userstoken';

import request from '@/utils/request.ts';
const store = useTokenStore();
const avatarstore = useUserAvatarStore();

const defaultValues = {
  account: '默认账号',
  password: '默认密码',
  email: 'example@example.com',
  phone: '1234567890',
  token: '默认Token',
  pet_name: 'test昵称',
  avatar: 'https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png'
};
const avatarUrl = ref(defaultValues.avatar);
const account = ref(defaultValues.account);
const password = ref(defaultValues.password);
const email = ref(defaultValues.email);
const phone = ref(defaultValues.phone);
const token = ref(defaultValues.token);
const nickname = ref(defaultValues.pet_name);
const activeName = ref('first');

const fetchPersonalPageData = async () => {
  try {
    const response = await request.post('/personal_page', { user: store.token.access_token });
    if (response.data.success === 'true' && response.data.data.length > 0) {
      const data = response.data.data[0];
      account.value = data.account ?? defaultValues.account;
      password.value = data.password ?? defaultValues.password;
      email.value = data.email ?? defaultValues.email;
      phone.value = data.phone ?? defaultValues.phone;
      token.value = data.token ?? defaultValues.token;
      nickname.value = data.pet_name ?? defaultValues.pet_name;
      avatarUrl.value = response.data.url;
      console.log('avatarUrl:', response.data.url);
    } else {
      ElMessage.error(`获取数据失败: ${response.data.message}`);
    }
  } catch (error) {
    const err = error as Error;
    ElMessage.error('发生错误: ' + err.message);
  }
};



const personClass = ref('');
const avatarClass = ref('');

const updatePersonClass = () => {
  if (window.innerWidth <= 1024) {
    personClass.value = 'person-small';
    avatarClass.value = 'avatar-small';
  } else {
    personClass.value = 'person-large';
    avatarClass.value = 'avatar-large';
  }
};

onMounted(() => {
  avatarstore.avatarUrl = avatarstore.avatarUrl || 'https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png';
  fetchPersonalPageData();
  updatePersonClass();
  window.addEventListener('resize', updatePersonClass);
});

const handleClick = (tab: TabsPaneContext, event: Event) => {
  console.log(tab, event);
};
</script>

<style>
html, body, #app, .custom-page-header {
  width: 100%;
  margin: 0;
  padding: 0;
}

.custom-page-header {
  padding: 16px;
  background-color: #f5f7fa;
  border-bottom: 1px solid #ebeef5;
  display: flex;
  flex-direction: column;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  position: relative;
  box-sizing: border-box;
}

.header-extra {
  display: flex;
  justify-content: space-between;
  width: 100%;
}

.button-container {
  display: flex;
  justify-content: flex-start;
}

.demo-tabs > .el-tabs__content {
  padding: 32px;
  color: #6b778c;
  font-size: 16px;
  font-weight: 400;
}

.tags {
  display: flex;
  align-items: center;
  flex-direction: column;
}

@media (min-width: 1024px) {
  .person-large {
    display: flex;
    align-items: flex-start;
    width: 100%;
  }
  .person-large .avatar-container {
    display: flex;
    align-items: center;
    margin-right: 20px;
  }
  .avatar-large {
    width: 150px;
    height: 150px;
  }
  .person-large .info-container {
    display: flex;
    flex-direction: column;
    margin-left: 50px;
  }
  .person-large .tags {
    display: flex;
    flex-direction: column;
    margin-bottom: 20px;
  }
}

@media (max-width: 1024px) {
  .person-small {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin-left: 0;
    width: 100%;
  }
  .person-small .header-extra {
    display: flex;
    justify-content: space-between;
    width: 100%;
    margin-bottom: 20px;
  }
  .person-small .avatar-container {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    margin-bottom: 20px;
    width: 100%;
  }
  .avatar-small {
    width: 100px;
    height: 100px;
  }
  .person-small .tags {
    display: flex;
    justify-content: flex-start;
    flex-wrap: wrap;
    margin-left: 10px;
  }
  .person-small .info-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
