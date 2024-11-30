<template>
  <transition name="slide" style="height:50%">
    <div v-if="isVisible" 
         class="sidebar fixed height-500 top-20 right-0 h-full bg-white shadow-lg border-l border-gray-300 z-50" 
         :style="{ width: sidebarWidth + 'px' }" 
         ref="sidebar">
      <div class="flex justify-between items-center bg-gray-100 p-2 border-b border-gray-300">
        <span>AI 助手</span>
        <button class="text-gray-500 hover:text-gray-800" @click="closeSidebar">&times;</button>
      </div>
      <div class="p-4 flex">
        <div class="flex flex-col ">
          <Chat v-if="ischatvisible" />
          <Translate v-if="istranslatevisible"/>
          <Ocr v-if="isocrvisible"/>
          <Write v-if="iswritevisible"/>
          <vtoc v-if="isvtocvisible"/>
        </div>
      
        <div class="flex flex-col fixed-right">
          <img src="@/assets/talk.svg" class="ocr" @click="showChat"/>
          <img src="@/assets/translate1.svg" class="ocr" @click="showTranslate"/>
          <img src="@/assets/ocr.svg" class="ocr" @click="showOcr"/>
          <img src="@/assets/write.svg" class="ocr" @click="showWrite"/>
          <img src="@/assets/语音转文字.svg" class="ocr" @click="showvtoc"/>
          <img src="@/assets/语音转文字.svg" class="ocr" @click="tobig"/>
        </div>
      </div>
    </div>
  </transition>
</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';
import Chat from "./Chat.vue"
import Ocr from "./Ocr.vue"
import Translate from "./Translate.vue"
import Write from "./Write.vue"
import vtoc from "./vtoc.vue"
import { useRouter } from 'vue-router';
const router = useRouter();
const ischatvisible = ref(true);
const istranslatevisible = ref(false);
const isocrvisible = ref(false);
const iswritevisible = ref(false);
const isvtocvisible = ref(false);
const pdfVisible = ref(false);
const pdfContent = ref(null);
const tobig=()=>{
  router.push('/Chat_rag')
  emit('close');
}
const props = defineProps({
  isVisible: Boolean
});

const showPdfContent = (content) => {
  pdfContent.value = content;
  pdfVisible.value = true;
};

const emit = defineEmits(['close']);
const sidebarWidth = 350; // 修改宽度为400px

const closeSidebar = () => {
  emit('close');
};

const showChat = () => {
  ischatvisible.value = true;
  istranslatevisible.value = false;
  isocrvisible.value = false;
  iswritevisible.value = false;
  isvtocvisible.value = false;
};

const showTranslate = () => {
  ischatvisible.value = false;
  istranslatevisible.value = true;
  isocrvisible.value = false;
  iswritevisible.value = false;
  isvtocvisible.value = false;
};

const showOcr = () => {
  ischatvisible.value = false;
  istranslatevisible.value = false;
  isocrvisible.value = true;
  iswritevisible.value = false;
  isvtocvisible.value = false;
};

const showWrite = () => {
  ischatvisible.value = false;
  istranslatevisible.value = false;
  isocrvisible.value = false;
  iswritevisible.value = true;
  isvtocvisible.value = false;
};

const showvtoc = () => {
  ischatvisible.value = false;
  istranslatevisible.value = false;
  isocrvisible.value = false;
  iswritevisible.value = false;
  isvtocvisible.value = true;
};
</script>

<style scoped>
@import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.0/dist/tailwind.min.css');

.fixed-right {
  position: fixed;
  right: 0;
  top: 10; /* 可以根据需要调整垂直位置 */
}

.sidebar {
  transition: transform 0.3s ease-in-out;
}

.slide-enter-active, .slide-leave-active {
  transition: transform 0.3s ease-in-out;
}

.slide-enter, .slide-leave-to {
  transform: translateX(100%);
}

.ocr {
  display: flex;
  align-items: center;
  text-decoration: none;
  color: black;
  height: 30px;
  margin-bottom: 20px;
  margin-right: -5px;
  
  img {
    width: 32px;
    height: 32px;
  }
}
</style>
