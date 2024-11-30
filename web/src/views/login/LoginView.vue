<script setup>
import { useLayout } from '@/layout/composables/layout';
import { ref, computed, reactive } from 'vue';
import { ElMessage } from 'element-plus';
import AppConfig from '@/layout/AppConfig.vue';
import { login } from '@/api/users';
import { useRouter } from 'vue-router';
import { useTokenStore,useUserAvatarStore } from '@/store/userstoken';

const { layoutConfig } = useLayout();
const checked = ref(false);
const store = useTokenStore();
const router = useRouter();
const isLoading = ref(false);
const formRef = ref(null); // 修改此处
const avatar=useUserAvatarStore();
const form = reactive({
  user: 'admin',
  password: '123456',
});

const logoUrl = computed(() => {
  return `/editFront1/layout/images/${layoutConfig.darkTheme.value ? 'logo-white' : 'logo-dark'}.svg`;
});

// 登录事件处理
const onSubmit = async () => {
  try {
    isLoading.value = true;
    await formRef.value?.validate();
    const res = await login(form);
    if (res.data.success == "false") {
      ElMessage.error(res.data.message);
      isLoading.value = false;
      return;
    }
    store.saveToken(res.data.content);
    ElMessage.success("登录成功");
    router.push('/');
  } catch (error) {
    ElMessage.error(error.message || "登录失败");
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="surface-ground flex align-items-center justify-content-center min-h-screen min-w-screen overflow-hidden">
    <div class="flex flex-column align-items-center justify-content-center">
      <img :src="logoUrl" alt="Sakai logo" class="mb-5 w-6rem flex-shrink-0" />
      <div style="border-radius: 56px; padding: 0.3rem; background: linear-gradient(180deg, var(--primary-color) 10%, rgba(33, 150, 243, 0) 30%)">
        <div class="w-full surface-card py-8 px-5 sm:px-8" style="border-radius: 53px">
          <div class="text-center mb-5">
            <img :src="`${avatar.avatarUrl}`" alt="Image" height="50" class="mb-3" />
            <div class="text-900 text-3xl font-medium mb-3">欢迎, {{ form.user }}</div>
            <span class="text-600 font-medium">登入</span>
          </div>

          <el-form :model="form" ref="formRef" label-width="auto">
            <div class="form-control">
              <label for="email1" class="block text-900 text-xl font-medium mb-2">账户</label>
              <InputText id="email1" type="text" placeholder="Account" class="w-full md:w-30rem mb-5" style="padding: 1rem" v-model="form.user" />
            </div>
            <div class="form-control">
              <label for="password1" class="block text-900 font-medium text-xl mb-2">密码</label>
              <Password id="password1" v-model="form.password" placeholder="输入密码" :toggleMask="true" class="w-full mb-3" inputClass="w-full" :inputStyle="{ padding: '1rem' }" />
            </div>
            <div class="flex align-items-center justify-content-between mb-5 gap-5">
              <div class="flex align-items-center">
                <Checkbox v-model="checked" id="rememberme1" binary class="mr-2" />
                <label for="rememberme1">记住我</label>
              </div>
              <a class="font-medium no-underline ml-2 text-right cursor-pointer" style="color: var(--primary-color)">忘记密码？</a>
            </div>
            <Button label="登录" class="w-full p-3 text-xl" :loading="isLoading" @click="onSubmit"></Button>
          </el-form>
          <a class="font-medium no-underline ml-2 text-right cursor-pointer" href="http://localhost:5173/register" style="color: var(--primary-color)">没有账户？点击注册</a>
        </div>
      </div>
    </div>
  </div>
  <AppConfig simple />
</template>

<style scoped>
.pi-eye {
  transform: scale(1.6);
  margin-right: 1rem;
}

.pi-eye-slash {
  transform: scale(1.6);
  margin-right: 1rem;
}
</style>
