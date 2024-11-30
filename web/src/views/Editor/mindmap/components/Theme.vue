<template>
  <div class="container">
    <h1
      style="
        font-size: 20px;
        border-bottom: 2px solid black;
        padding-bottom: 10px;
      "
    >
      选择变更的主题模板
    </h1>
    <input
      type="text"
      v-model="searchText"
      placeholder="搜索主题"
      class="search-input"
    />
    <div class="icons">
      <div
        class="icon-item"
        v-for="(item, index) in filteredThemes"
        :key="index"
        @click="selectTheme(item.name)"
      >
        <el-icon style="width: 200px; height: 64px"
          ><img :src="item.img"
        /></el-icon>
        <div class="icon-text">{{ item.name }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, defineEmits,computed } from "vue";

const themelist = ref([
  {
    name: "autumn",
    img: new URL("@/assets/img/themes/autumn.jpg", import.meta.url).href,
  },
  {
    name: "dark",
    img: new URL("@/assets/img/themes/dark.jpg", import.meta.url).href,
  },
  {
    name: "classic",
    img: new URL("@/assets/img/themes/classic.jpg", import.meta.url).href,
  },
  {
    name: "classic2",
    img: new URL("@/assets/img/themes/classic2.jpg", import.meta.url).href,
  },
  {
    name: "classic3",
    img: new URL("@/assets/img/themes/classic3.jpg", import.meta.url).href,
  },
  {
    name: "classicBlue",
    img: new URL("@/assets/img/themes/classicBlue.jpg", import.meta.url).href,
  },
  {
    name: "classicGreen",
    img: new URL("@/assets/img/themes/classicGreen.jpg", import.meta.url).href,
  },
  {
    name: "earthYellow",
    img: new URL("@/assets/img/themes/earthYellow.jpg", import.meta.url).href,
  },
  {
    name: "coffee",
    img: new URL("@/assets/img/themes/coffee.jpg", import.meta.url).href,
  },
]);

const searchText=ref("")
const filteredThemes = computed(() => {
  if (!searchText.value) return themelist.value;
  return themelist.value.filter((theme) =>
    theme.name.toLowerCase().includes(searchText.value.toLowerCase())
  );
});

const emit = defineEmits(["theme-selected"]);
const selectTheme = (themeName: string) => {
  emit("theme-selected", themeName);
};
</script>

<style scoped>
/* 搜索框样式 */
.search-input {
  width: 100%;
  padding: 8px 12px;
  margin-bottom: 20px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.container {
  border: 2px solid black;
  box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.1); /* Add a shadow effect */
  height: 500px;
  overflow-y: auto;
  padding: 20px; /* Add padding for better spacing */
}

.icons {
  display: flex;
  gap: 20px; /* Adjust the spacing between icons */
  justify-content: center; /* Center the icons */
  flex-wrap: wrap;
}

.icon-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  cursor: pointer;
}

.icon-text {
  margin-top: 5px;
  font-size: 14px;
  text-align: center;
}
</style>
