<script setup>
import { ref } from 'vue';
import AppMenuItem from '@/layout/AppMenuItem.vue';

// 定义图标路径
const pptIcon = new URL('@/assets/PPT2.svg', import.meta.url).href;
const pdfIcon = new URL('@/assets/PDF3.svg', import.meta.url).href;
const mindmapIcon = new URL('@/assets/mind-map.svg', import.meta.url).href;
const wordIcon = new URL('@/assets/Word2.svg', import.meta.url).href;
const flowchartIcon = new URL('@/assets/流程图.svg', import.meta.url).href;
const model = ref([
    {
        label: 'Home',
        items: [{ label: 'Dashboard', icon: 'pi pi-fw pi-home', to: '/dashboard' }]
    },
    {
        label: '开始',
        items: [
            {
                label: '创建文档',
                icon: 'pi pi-fw pi-file',
                items: [
                    {
                        label: 'word',
                        icon: wordIcon,
                        to: '/doc'
                    },
                    {
                        label: 'mindmap',
                        icon: mindmapIcon,
                        to: '/mindmap'
                    },
                    {
                        label: 'flowchart',
                        icon: flowchartIcon,
                        to: '/flowchart'
                    },
                    {
                        label: 'paint',
                        icon: 'pi pi-fw pi-pencil',
                        to: '/drawing'
                    },
                ]
            },
            {
                label:'知识库',
                icon: 'pi pi-fw pi-file',
                to: '/knowledge'
            },
            {
                label:'AI',
                icon: 'pi pi-fw pi-file',
                to: '/chat_rag'
            }
        ]
    }
]);
</script>

<template>
    <ul class="layout-menu">
        <template v-for="(item, i) in model" :key="item.label">
            <app-menu-item v-if="!item.separator" :item="item" :index="i"></app-menu-item>
            <li v-if="item.separator" class="menu-separator"></li>
        </template>
    </ul>
</template>

<style lang="scss" scoped></style>
