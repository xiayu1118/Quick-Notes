<template>
  <div style="height: 500px; width: 100%;">
    <RenderReactNode :node="excalidrawNode" />
  </div>
</template>

<script setup>
import { getReactNode, RenderReactNode } from 'veaury';
import ExcalidrawWrapper from '@/react_app/Basic.jsx';

// 获取React组件节点
const excalidrawNode = getReactNode(ExcalidrawWrapper);
</script>

<style scoped>
/* 你的样式 */
</style>
