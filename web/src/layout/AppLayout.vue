<script setup>
import { computed, watch, ref } from 'vue';
import AppTopbar from '@/layout/AppTopbar.vue';
import AppFooter from '@/layout/AppFooter.vue';
import AppSidebar from '@/layout/AppSidebar.vue';
import AppConfig from '@/layout/AppConfig.vue';
import { useLayout } from '@/layout/composables/layout.js';
import AiSidebar from "@/components/AI/AiSidebar.vue";
const { layoutConfig, layoutState, isSidebarActive } = useLayout();
 // 用于存储PDF文件的URL
const outsideClickListener = ref(null);

const isSidebarVisible = ref(false);

const toggleSidebar = () => {
  isSidebarVisible.value = !isSidebarVisible.value;
};

const closeSidebar = () => {
  isSidebarVisible.value = false;
};
watch(isSidebarActive, (newVal) => {
    if (newVal) {
        bindOutsideClickListener();
    } else {
        unbindOutsideClickListener();
    }
});

const containerClass = computed(() => {
    return {
        'layout-theme-light': layoutConfig.darkTheme.value === 'light',
        'layout-theme-dark': layoutConfig.darkTheme.value === 'dark',
        'layout-overlay': layoutConfig.menuMode.value === 'overlay',
        'layout-static': layoutConfig.menuMode.value === 'static',
        'layout-static-inactive': layoutState.staticMenuDesktopInactive.value && layoutConfig.menuMode.value === 'static',
        'layout-overlay-active': layoutState.overlayMenuActive.value,
        'layout-mobile-active': layoutState.staticMenuMobileActive.value,
        'p-ripple-disabled': layoutConfig.ripple.value === false
    };
});
const bindOutsideClickListener = () => {
    if (!outsideClickListener.value) {
        outsideClickListener.value = (event) => {
            if (isOutsideClicked(event)) {
                layoutState.overlayMenuActive.value = false;
                layoutState.staticMenuMobileActive.value = false;
                layoutState.menuHoverActive.value = false;
            }
        };
        document.addEventListener('click', outsideClickListener.value);
    }
};
const unbindOutsideClickListener = () => {
    if (outsideClickListener.value) {
        document.removeEventListener('click', outsideClickListener);
        outsideClickListener.value = null;
    }
};
const isOutsideClicked = (event) => {
    const sidebarEl = document.querySelector('.layout-sidebar');
    const topbarEl = document.querySelector('.layout-menu-button');

    return !(sidebarEl.isSameNode(event.target) || sidebarEl.contains(event.target) || topbarEl.isSameNode(event.target) || topbarEl.contains(event.target));
};

</script>

<template>
    <div class="layout-wrapper" :class="containerClass">
        <app-topbar></app-topbar>
        <div class="layout-sidebar">
            <app-sidebar></app-sidebar>
        </div>
        
        <div class="layout-main-container">
            <div class="layout-main">
                <router-view></router-view>
            </div>
            <app-footer></app-footer>
            
        </div>
        
        <AiSidebar :isVisible="isSidebarVisible" @close="closeSidebar" />
    </div>

    <div type="primary" @click="toggleSidebar" class="toggle-button" >
      <img src="@/assets/ai.svg" class="logo" v-show="!isSidebarVisible"/>
    </div>
    
    <Toast />
</template>

<style lang="scss" scoped>
    .toggle-button {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 100;
}
// 新增PDF容器的样式

.logo {
  display: flex;
  align-items: center;
  text-decoration: none;
  justify-content: center;
  color: black;
  height: 40px;
  img {
    width: 20px;
    height: 20px;
  }
}
</style>