import exampleData from 'simple-mind-map/example/exampleData'
import { simpleDeepClone } from 'simple-mind-map/src/utils/index'

import bus from '@/utils/bus.js'

const SIMPLE_MIND_MAP_DATA = 'SIMPLE_MIND_MAP_DATA'
const SIMPLE_MIND_MAP_LANG = 'SIMPLE_MIND_MAP_LANG'
const SIMPLE_MIND_MAP_LOCAL_CONFIG = 'SIMPLE_MIND_MAP_LOCAL_CONFIG'

/**
 * 
 * @Desc: 克隆思维导图数据，去除激活状态
 */
const copyMindMapTreeData = (tree: { data?: any; children?: any }, root: { data: any; children: any[] }) => {
  tree.data = simpleDeepClone(root.data)
  // tree.data.isActive = false
  tree.children = []
  if (root.children && root.children.length > 0) {
    root.children.forEach((item, index) => {
      tree.children[index] = copyMindMapTreeData({}, item)
    })
  }
  return tree
}

/**
 * 
 * @Desc: 获取缓存的思维导图数据
 */
export const getData = () => {
  let store = localStorage.getItem(SIMPLE_MIND_MAP_DATA)
  if (store === null) {
    return simpleDeepClone(exampleData)
  } else {
    try {
      return JSON.parse(store)
    } catch (error) {
      return simpleDeepClone(exampleData)
    }
  }
}

/**
 * 
 * @Desc: 存储思维导图数据
 */
export const storeData = (data: { data: any; children: any[] }) => {
  try {
    let originData = getData()
    originData.root = copyMindMapTreeData({}, data)
    bus.emit('write_local_file', originData)
    let dataStr = JSON.stringify(originData)
    localStorage.setItem(SIMPLE_MIND_MAP_DATA, dataStr)
  } catch (error) {
    console.log(error)
  }
}

/**
 * 
 * @Desc: 存储思维导图配置数据
 */
export const storeConfig = (config: any) => {
  try {
    let originData = getData()
    originData = {
      ...originData,
      ...config
    }
    bus.emit('write_local_file', originData)
    let dataStr = JSON.stringify(originData)
    localStorage.setItem(SIMPLE_MIND_MAP_DATA, dataStr)
  } catch (error) {
    console.log(error)
  }
}

/**
 * 
 * @Desc: 存储语言
 */
export const storeLang = (lang: string) => {
  localStorage.setItem(SIMPLE_MIND_MAP_LANG, lang)
}

/**
 * 
 * @Desc: 获取存储的语言
 */
export const getLang = () => {
  let lang = localStorage.getItem(SIMPLE_MIND_MAP_LANG)
  if (lang) {
    return lang
  }
  storeLang('zh')
  return 'zh'
}

/**
 * 
 * @Desc: 存储本地配置
 */
export const storeLocalConfig = (config: { isZenMode: boolean; openNodeRichText: boolean; useLeftKeySelectionRightKeyDrag: boolean; isShowScrollbar: boolean }) => {
  localStorage.setItem(SIMPLE_MIND_MAP_LOCAL_CONFIG, JSON.stringify(config))
}

/**
 * 
 * @Desc: 获取本地配置
 */
export const getLocalConfig = () => {
  let config = localStorage.getItem(SIMPLE_MIND_MAP_LOCAL_CONFIG)
  if (config) {
    return JSON.parse(config)
  }
  return null
}