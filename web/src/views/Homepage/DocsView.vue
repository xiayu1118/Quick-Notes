<template>
  <el-card style="width: auto">
     <template #header>
     </template>
     
     <el-table :data="alldocs" border style="width: auto;display: flex ;">
       <el-table-column prop="date" label="修改日期" width="180" align="center" />
       <el-table-column prop="name" label="名称" width="400" align="center" />
       <el-table-column prop="sorts" label="类型" align="center" />
       <el-table-column prop="address" label="操作" align="center">
         <template #default="{ row }">
           <el-button @click="editDocument(row.name)">编辑</el-button>
           <el-button>删除</el-button>
         </template>
       </el-table-column>
     </el-table>
   </el-card>
 </template>
 
 <script setup lang="ts">
 import { onMounted, ref } from 'vue';

 import { useRouter } from 'vue-router';
 import { useTokenStore } from '@/store/userstoken';
import requestfile from '@/utils/requestfile.ts';
 const router1 = useRouter();
 const store = useTokenStore();
 
 const alldocs = ref([]);
 const fetchdocsdata = async () => {
   try {
     const response = await requestfile.post('/show_file', { user: store.token.access_token });
     if (response.data.success === 'true' && response.data.data.length > 0) {
       alldocs.value = response.data.data;
     } else {
       console.log(`获取数据失败: ${response.data.message}`);
     }
   } catch (error) {
    const err = error as Error;
    console.log('发生错误: ' + err.message);
   }
 };

 const editDocument = (name:string) => {
   router1.push({ name: 'wangEditor', query: { documentName: name } });
 };
 
 onMounted(() => {
   fetchdocsdata();
 });
 </script>
 
 <style lang="scss">
 </style>
 