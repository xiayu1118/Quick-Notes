<template>
    <nav class="breadcrumb">
      <span v-for="(crumb, index) in crumbs" :key="index">
        <router-link :to="crumb.path" v-if="index !== crumbs.length - 1">{{ crumb.name }}</router-link>
        <span v-else>{{ crumb.name }}</span>
        <span v-if="index !== crumbs.length - 1"> / </span>
      </span>
      <button @click="closePage" class="close-btn">x</button>
    </nav>
  </template>
  
  <script setup lang="ts">
  import { useRoute, useRouter } from 'vue-router';
  import { computed } from 'vue';
  
  const route = useRoute();
  const router = useRouter();
  
  const crumbs = computed(() => {
    let pathArray = route.path.split('/').filter(Boolean);
    let crumbArray = pathArray.map((path, index) => {
      return {
        name: path,
        path: '/' + pathArray.slice(0, index + 1).join('/')
      };
    });
    return crumbArray;
  });
  
  const closePage = () => {
    router.back();
  };
  </script>
  
  <style lang="scss" scoped>
  .breadcrumb {
    display: flex;
    align-items: center;
    font-size: 14px;
    margin-bottom: 16px;
  }
  
  .breadcrumb span {
    margin: 0 4px;
  }
  
  .close-btn {
    background: none;
    border: none;
    color: red;
    cursor: pointer;
    margin-left: 8px;
  }
  </style>
  