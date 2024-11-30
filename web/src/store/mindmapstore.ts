import { defineStore } from 'pinia'
import exampleData from 'simple-mind-map/example/exampleData'
import { storeLocalConfig } from '@/api/mindmap'

interface LocalConfig {
  isZenMode: boolean;
  openNodeRichText: boolean;
  useLeftKeySelectionRightKeyDrag: boolean;
  isShowScrollbar: boolean;
}

interface State {
  mindMapData: any | null;
  isHandleLocalFile: boolean;
  localConfig: LocalConfig;
  activeSidebar: string;
  isDark: boolean;
  isOutlineEdit: boolean;
  isReadonly: boolean;
}

export const useStore = defineStore('main', {
  state: (): State => ({
    mindMapData: null, // 思维导图数据
    isHandleLocalFile: false, // 是否操作的是本地文件
    localConfig: {
      // 本地配置
      isZenMode: false, // 是否是禅模式
      openNodeRichText: true, // 是否开启节点富文本
      useLeftKeySelectionRightKeyDrag: false, // 鼠标行为
      isShowScrollbar: false // 是否显示滚动条
    },
    activeSidebar: '', // 当前显示的侧边栏
    isDark: false, // 是否是暗黑模式
    isOutlineEdit: false, // 是否是大纲编辑模式
    isReadonly: false // 是否只读
  }),
  actions: {
    /**
     * 
     * @Desc: 设置思维导图数据
     */
    setMindMapData(data: any) {
      this.mindMapData = data;
    },
    /**
     * 
     * @Desc: 设置操作本地文件标志位
     */
    setIsHandleLocalFile(data: boolean) {
      this.isHandleLocalFile = data;
    },
    /**
     * 
     * @Desc: 设置本地配置
     */
    setLocalConfig(data: Partial<LocalConfig>) {
      this.localConfig = {
        ...this.localConfig,
        ...data
      };
      storeLocalConfig(this.localConfig);
    },
    /**
     * 
     * @Desc: 设置当前显示的侧边栏
     */
    setActiveSidebar(data: string) {
      this.activeSidebar = data;
    },
    /**
     * 
     * @Desc: 设置暗黑模式
     */
    setIsDark(data: boolean) {
      this.isDark = data;
    },
    /**
     * 
     * @Desc: 设置大纲编辑模式
     */
    setIsOutlineEdit(data: boolean) {
      this.isOutlineEdit = data;
    },
    /**
     * 
     * @Desc: 设置是否只读
     */
    setIsReadonly(data: boolean) {
      this.isReadonly = data;
    },
    /**
     * 
     * @Desc: 设置初始思维导图数据
     */
    async getUserMindMapData() {
      try {
        let { data } = {
          data: {
            data: {
              mindMapData: exampleData
            }
          }
        };
        this.setMindMapData(data.data.mindMapData);
      } catch (error) {
        console.log(error);
      }
    }
  }
});
