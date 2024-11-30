<script setup>
import { ref, watch, onMounted, shallowRef } from "vue";
import MindMap from "simple-mind-map";
import THeme from "@/views/Editor/mindmap/components/Theme.vue";
import Export from 'simple-mind-map/src/plugins/Export.js'
import ExportXMind from 'simple-mind-map/src/plugins/ExportXMind.js'
import ExportPDF from 'simple-mind-map/src/plugins/ExportPDF.js'
import requestdb from "@/utils/requestdb";   
const words=ref('')                                                                                
MindMap.usePlugin(ExportPDF)
MindMap.usePlugin(Export)
MindMap.usePlugin(ExportXMind)
let mindMap = null;
//记录前进回退
const isStart = ref(true);
const isEnd = ref(true);
// 当前激活的节点列表
const activeNodes = shallowRef([]);
const isOpenFreeDrag = ref(false);
//当前右键点击的类型
const type = ref("");
// 如果点击的节点，那么代表被点击的节点
const currentNode = shallowRef(null);
//菜单显示的位置
const left = ref(0);
const top = ref(0);
const show = ref(false);
// 记录鼠标右键按下的位置
const mousedownX = ref(0);
const mousedownY = ref(0);
const isMousedown = ref(false);

// 切换是否开启自由拖拽
const toggleFreeDrag = () => {
  isOpenFreeDrag.value = !isOpenFreeDrag.value;
  mindMap.updateConfig({
    enableFreeDrag: isOpenFreeDrag.value,
  });
};
// 恢复默认布局
const resetLayout = () => {
  mindMap.execCommand("RESET_LAYOUT");
};
// 回退
const back = () => {
  mindMap.execCommand("BACK");
};

// 前进
const forward = () => {
  mindMap.execCommand("FORWARD");
};

// 插入兄弟节点
const insertNode = () => {
  mindMap.execCommand("INSERT_NODE");
};

// 插入子节点
const insertChildNode = () => {
  mindMap.execCommand("INSERT_CHILD_NODE");
};

// 删除节点
const deleteNode = () => {
  mindMap.execCommand("REMOVE_NODE");
};

// 插入图片
const inserImage = () => {
  activeNodes.value.forEach((node) => {
    node.setImage({
      url: "https://lxqnsys.oss-cn-beijing.aliyuncs.com/qlx/xh2AXkBxYm5jGe5fD7DWYrC5b.png",
      title: "图片的标题或描述",
      width: 100, // 图片的宽高也不能少
      height: 100,
    });
  });
};

const cut = () => {
  mindMap.renderer.cut();
};
const copy = () => {
  mindMap.renderer.copy();
};
const paste = () => {
  mindMap.renderer.paste();
};

// 插入图标
const insertIcon = () => {
  const iconList = ["priority_1", "priority_2"];
  activeNodes.value.forEach((node) => {
    node.setIcon(iconList);
  });
};

// 插入超链接
const insertLink = () => {
  activeNodes.value.forEach((node) => {
    node.setHyperlink("http://lxqnsys.com/", "理想青年实验室");
  });
};

// 插入备注
const insertNote = () => {
  activeNodes.value.forEach((node) => {
    node.setNote("备注内容");
  });
};

// 插入标签
const insertTag = () => {
  activeNodes.value.forEach((node) => {
    node.setTag(["标签1", "标签2"]);
  });
};

// 插入概要
const insertGa = () => {
  mindMap.execCommand("ADD_GENERALIZATION", {
    text: "自定义概要内容",
  });
};

// 插入关联线
const insertLine = () => {
  mindMap.associativeLine.createLineFromActiveNode();
};
const Theme = ref("");
// 切换主题
const changeTheme = (themeName) => {
  mindMap.setTheme(themeName);
};

// 注册并使用新主题
const defineTheme = () => {
  MindMap.defineTheme("redSpirit", {
    // 背景颜色
    backgroundColor: "rgb(255, 238, 228)",
    // 连线的颜色
    lineColor: "rgb(230, 138, 131)",
    lineWidth: 3,
    // 概要连线的粗细
    generalizationLineWidth: 3,
    // 概要连线的颜色
    generalizationLineColor: "rgb(222, 101, 85)",
    // 根节点样式
    root: {
      fillColor: "rgb(207, 44, 44)",
      color: "rgb(255, 233, 157)",
      borderColor: "",
      borderWidth: 0,
      fontSize: 24,
      active: {
        borderColor: "rgb(255, 233, 157)",
        borderWidth: 3,
      },
    },
    // 二级节点样式
    second: {
      fillColor: "rgb(255, 255, 255)",
      color: "rgb(211, 58, 21)",
      borderColor: "rgb(222, 101, 85)",
      borderWidth: 2,
      fontSize: 18,
      active: {
        borderColor: "rgb(255, 233, 157)",
      },
    },
    // 三级及以下节点样式
    node: {
      fontSize: 14,
      color: "rgb(144, 71, 43)",
      active: {
        borderColor: "rgb(255, 233, 157)",
      },
    },
    // 概要节点样式
    generalization: {
      fontSize: 14,
      fillColor: "rgb(255, 247, 211)",
      borderColor: "rgb(255, 202, 162)",
      borderWidth: 2,
      color: "rgb(187, 101, 69)",
      active: {
        borderColor: "rgb(222, 101, 85)",
      },
    },
  });
  mindMap.setTheme("redSpirit");
};

const fontFamily = ref("");
const fontSize = ref(16);
const lineHeight = ref(1.5);
const color = ref("");
const textDecoration = ref("");

// 设置文字样式
const setTextStyle = () => {
  activeNodes.value.forEach((node) => {
    node.setStyle("fontFamily", fontFamily.value);
    node.setStyle("lineHeight", lineHeight.value);
    node.setStyle("fontSize", fontSize.value);
    node.setStyle("color", color.value);
    node.setStyle("textDecoration", textDecoration.value);
  });
};
const setbold = () => {
  activeNodes.value.forEach((node) => {
    node.setStyle("fontWeight", "bold");
  });
};
//字体格式选项
const fontFamilyOptions = [
  {
    label: "宋体",
    value: "宋体, SimSun, Songti SC",
  },
  {
    label: "黑体",
    value: "SimHei",
  },
  {
    label: "楷体",
    value: "KaiTi",
  },
  {
    label: "仿宋",
    value: "FangSong",
  },
  {
    label: "微软雅黑",
    value: "Microsoft YaHei",
  },
  {
    label: "Arial",
    value: "Arial",
  },
  {
    label: "Times New Roman",
    value: "Times New Roman",
  },
  {
    label: "Georgia",
    value: "Georgia",
  },
];
//字体大小选项
const fontSizeOptions = [
  {
    label: "2px",
    value: 2,
  },
  {
    label: "4px",
    value: "4",
  },
  {
    label: "6px",
    value: 6,
  },
  {
    label: "8px",
    value: 8,
  },
  {
    label: "10px",
    value: 10,
  },
  {
    label: "12px",
    value: 12,
  },
  {
    label: "14px",
    value: 14,
  },
  {
    label: "16px",
    value: 16,
  },
  {
    label: "18px",
    value: 18,
  },
  {
    label: "20px",
    value: 20,
  },
];
//行高选项
const lineHeightOptions = [
  {
    label: "1",
    value: 1,
  },
  {
    label: "1.5",
    value: 1.5,
  },
  {
    label: "2",
    value: 2,
  },
  {
    label: "2.5",
    value: 2.5,
  },
  {
    label: "3",
    value: 3,
  },
];
const textDecorationOptions = [
  {
    label: "none",
    value: "none",
  },
  {
    label: "下划线",
    value: "underline",
  },
  {
    label: "删除线",
    value: "line-through",
  },
  {
    label: "上划线",
    value: "overline",
  },
];
watch([fontFamily, fontSize, color, lineHeight, textDecoration], () => {
  setTextStyle();
});

const borderColor = ref("");
const borderDasharray = ref("");
const borderWidth = ref("");
const borderRadius = ref("");
// 设置边框样式
const setBorderStyle = () => {
  activeNodes.value.forEach((node) => {
    node.setStyle("borderColor", borderColor.value);
    node.setStyle("borderDasharray", borderDasharray.value);
    node.setStyle("borderWidth", borderWidth.value);
    node.setStyle("borderRadius", borderRadius.value);
  });
};
watch([borderColor, borderDasharray, borderWidth, borderRadius], () => {
  setBorderStyle();
});

const fillColor = ref("");
// 设置背景
const setBg = () => {
  activeNodes.value.forEach((node) => {
    node.setStyle("fillColor", fillColor.value);
  });
};
const shape = ref("");
// 设置形状
const setShape = () => {
  activeNodes.value.forEach((node) => {
    node.setStyle("shape", shape.value);
  });
};

// 设置线条样式
const setLineStyle = () => {
  activeNodes.value.forEach((node) => {
    node.setStyle("lineColor", "#000");
    node.setStyle("lineDasharray", "5, 5, 1, 5");
    node.setStyle("lineWidth", 3);
  });
};
const paddingX = ref(0);
const paddingY = ref(0);
// 设置内边距
const setPadding = () => {
  activeNodes.value.forEach((node) => {
    node.setStyle("paddingX", paddingX.value);
    node.setStyle("paddingY", paddingY.value);
  });
};
const Layout = ref("");
const changeLayout = () => {
  mindMap.setLayout("oganizationStructure");
};

const hideMenu = () => {
  show.value = false;
};

// 导入文件
const onInput = (e) => {
  let file = e.target.files[0]
  let reg = /\.(smm|xmind|json|xlsx|md)$/
  if (!reg.test(file.name)) {
    alert('请选择.smm、.json、.xmind、.xlsx、.md文件')
    return
  }
  if (/\.(smm|json)$/.test(file.name)) {
    handleSmm(file)
  } else if (/\.xmind$/.test(file.name)) {
    handleXmind(file)
  } else if (/\.xlsx$/.test(file.name)) {
    handleExcel(file)
  } else if (/\.md$/.test(file.name)) {
    handleMd(file)
  }
}

// 处理.smm、.json文件
const handleSmm = (file) => {
  let fileReader = new FileReader()
  fileReader.readAsText(file)
  fileReader.onload = evt => {
    try {
      let data = JSON.parse(evt.target.result)
      if (typeof data !== 'object') {
        throw new Error('文件内容有误')
      }
      setData(data)
    } catch (error) {
      console.log(error)
    }
  }
}

// 处理.xmind文件
const handleXmind = async (file) => {
  try {
    let data = await MindMap.xmind.parseXmindFile(file)
    setData(data)
  } catch (error) {
    console.log(error)
  }
}

// 处理.xlsx文件
const handleExcel = async (file) => {
  // 略
}

// 处理markdown文件
const handleMd = async (file) => {
  let fileReader = new FileReader()
  fileReader.readAsText(file)
  fileReader.onload = async evt => {
    try {
      let data = await MindMap.markdown.transformMarkdownTo(evt.target.result)
      setData(data)
    } catch (error) {
      console.log(error)
    }
  }
}

// 导入数据
const setData = (data) => {
  if (data.root) {
    mindMap.setFullData(data)
  } else {
    mindMap.setData(data)
  }
  mindMap.view.reset()
}


const Exportpng = () => {
  // 导出为PNG
mindMap.doExport.png().then((data) => {
  // 创建下载链接并触发下载
  let a = document.createElement('a')
  a.href = data
  a.download = 'filename.png'
  a.click()
})
};
const Exportjson = () => {
  mindMap.doExport.json().then((data) => {
  // 创建下载链接并触发下载
  let a = document.createElement('a')
  a.href = data
  a.download = 'filename.json'
  a.click()
})
};
const Exportxmind = () => {
  mindMap.doExport.xmind().then((data) => {
  // 创建下载链接并触发下载
  let a = document.createElement('a')
  a.href = data
  a.download = 'filename.xmind'
  a.click()
})
};
const Exporttxt = () => {
  mindMap.doExport.txt().then((data) => {
  // 创建下载链接并触发下载
  let a = document.createElement('a')
  a.href = data
  a.download = 'filename.txt'
  a.click()
})
};

const Exportpdf = () => {
  mindMap.doExport.pdf().then((data) => {
  // 创建下载链接并触发下载
  let a = document.createElement('a')
  a.href = data
  a.download = 'filename.pdf'
  a.click()
})
};
import requestai from "@/utils/requestai";
const oneByOneCreateAll = async() => {
  console.log("123")
  // const response=requestai.post('/mindmap_change',{words:"test"})
  //  console.log(response.data)
  const dataList = [
    {
      data: {
        text: '根节点',
        uid: '1'
      },
      children: []
    },
    {
      data: {
        text: '根节点',
        uid: '1'
      },
      children: [
        {
          data: {
            text: '二级节点',
            uid: '2'
          },
          children: []
        }
      ]
    },
    {
      data: {
        text: '根节点',
        uid: '1'
      },
      children: [
        {
          data: {
            text: '二级节点',
            uid: '2'
          },
          children: []
        },
        {
          data: {
            text: '二级节点',
            uid: '3'
          },
          children: []
        }
      ]
    },
    {
      data: {
        text: '根节点',
        uid: '1'
      },
      children: [
        {
          data: {
            text: '二级节点',
            uid: '2'
          },
          children: [
            {
              data: {
                text: '三级节点',
                uid: '4'
              },
              children: []
            }
          ]
        },
        {
          data: {
            text: '二级节点',
            uid: '3'
          },
          children: []
        }
      ]
    }
  ]
  
  let index = 0
  const run = () => {
    if (index >= dataList.length) return
    const cur = dataList[index]
    if (index === 0) {
      mindMap.setData(cur)
      let waitUid = ''
      mindMap.on('data_change_detail', (list) => {
        const lastCreate = list.filter((item) => {
          return item.action === 'create'
        })[0]
        if (lastCreate) {
          const uid = lastCreate.data.data.uid
          const node = mindMap.renderer.findNodeByUid(uid)
          if (node) {
            mindMap.renderer.moveNodeToCenter(node)
          } else {
            waitUid = uid
          }
        }
      })
      mindMap.on('node_tree_render_end', () =>{
        if (waitUid) {
          waitUid = ''
          const node = mindMap.renderer.findNodeByUid(waitUid)
          if (node) {
            mindMap.renderer.moveNodeToCenter(node)
          }
        }
      })
    } else {
      // mindMap.updateData(cur)
      mindMap.renderer.setData(cur)
      mindMap.render()
      mindMap.command.addHistory()
    }
    setTimeout(() => {
      index++
      run()
    }, 1000)
  }
  run()
}

onMounted(() => {
  mindMap = new MindMap({
    el: document.getElementById("mindMapContainer"),
    data: {
      data: {
        text: "根节点",
      },
      children: [
        {
          data: {
            text: "二级节点",
          },
          children: [],
        },
        {
          data: {
            text: "二级节点",
          },
          children: [],
        },
      ],
    },

    initRootNodePosition: ["left", "center"],
  });
  //监听右键点击事件
  mindMap.on("node_contextmenu", (e, node) => {
    type.value = "node";
    left.value = e.clientX + 10;
    top.value = e.clientY + 10;
    show.value = true;
    currentNode.value = node;
  });
  //监听右键菜单取消事件
  mindMap.on("svg_mousedown", (e) => {
    // 如果不是右键点击直接返回
    if (e.which !== 3) {
      return;
    }
    mousedownX.value = e.clientX;
    mousedownY.value = e.clientY;
    isMousedown.value = true;
  });

  mindMap.on("data_change", (data) => {
    // data数据是不带节点对象的纯数据
    // 如果你需要操作节点对象，可以使用mindMap.renderer.renderTree
    console.log(data, mindMap.renderer.renderTree);
  });

  mindMap.on("mouseup", (e) => {
    if (!isMousedown.value) {
      return;
    }
    isMousedown.value = false;
    // 如果鼠标松开和按下的距离大于3，则不认为是点击事件
    if (
      Math.abs(mousedownX.value - e.clientX) > 3 ||
      Math.abs(mousedownX.value - e.clientY) > 3
    ) {
      hideMenu();
      return;
    }
    type.value = "svg";
    left.value = e.clientX + 10;
    top.value = e.clientY + 10;
    show.value = true;
  });

  // 监听节点激活事件
  mindMap.on("node_active", (node, nodeList) => {
    activeNodes.value = nodeList;
  });

  //前进回退事件
  mindMap.on("back_forward", (index, len) => {
    isStart.value = index <= 0;
    isEnd.value = index >= len - 1;
  });
});
</script>

<template>
  <div class="mindMapDemo">
    <div id="mindMapContainer"></div>

    <div class="toolbar">

      <el-popover
    placement="bottom"
    title="描述你想生成的样式"
    :width="200"
    trigger="click"
    content="this is content, this is content, this is content"
  >
    <template #reference>
      <el-button type="primary" @click.native.prevent>
      AI
    </el-button>
    </template>
    <div>
    <el-input placeholder="请输入内容"   @keyup.enter="oneByOneCreateAll" v-model="words"></el-input>
  </div>
  </el-popover>

      <div class="toolbar-item" @click="back" v-if="!isStart">
        <el-icon><img src="@/assets/回退.svg" alt="切换主题" /></el-icon>
        <span>回退</span>
      </div>
      <div class="toolbar-item" @click="forward" v-if="!isEnd">
        <el-icon><img src="@/assets/前进.svg" alt="切换主题" /></el-icon>
        <span>前进</span>
      </div>
      <div class="toolbar-item" @click="insertNode">
        <el-icon><img src="@/assets/兄弟节点.svg" alt="切换主题" /></el-icon>
        <span>插入兄弟节点</span>
      </div>
      <div class="toolbar-item" @click="insertChildNode">
        <el-icon><img src="@/assets/添加子节点.svg" alt="切换主题" /></el-icon>
        <span>插入子节点</span>
      </div>
      <div class="toolbar-item" @click="deleteNode">
        <el-icon><img src="@/assets/删除节点.svg" alt="切换主题" /></el-icon>
        <span>删除节点</span>
      </div>

      <div class="toolbar-item" @click="inserImage">
        <el-icon><img src="@/assets/图片.svg" alt="插入图片" /></el-icon>
        <span>图片</span>
      </div>
      <div class="toolbar-item" @click="insertIcon">
        <el-icon><img src="@/assets/笑脸.svg" alt="插入图标" /></el-icon>
        <span>图标</span>
      </div>
      <div class="toolbar-item" @click="insertLink">
        <el-icon><img src="@/assets/链接.svg" alt="插入超链接" /></el-icon>
        <span>超链接</span>
      </div>
      <div class="toolbar-item" @click="insertNote">
        <el-icon><img src="@/assets/操作-备注.svg" alt="插入备注" /></el-icon>
        <span>备注</span>
      </div>
      <div class="toolbar-item" @click="insertTag">
        <el-icon><img src="@/assets/插入-标签.svg" alt="插入标签" /></el-icon>
        <span>插入标签</span>
      </div>
      <div class="toolbar-item" @click="insertGa">
        <el-icon><img src="@/assets/概要.svg" alt="插入概要" /></el-icon>
        <span>插入概要</span>
      </div>
      <div class="toolbar-item" @click="insertLine">
        <el-icon><img src="@/assets/全科.svg" alt="插入关联线" /></el-icon>
        <span>插入关联线</span>
      </div>
    </div>

    <div class="toolbar-2">
      <div class="toolbar-item" @click="changeLayout">
        <el-icon><img src="@/assets/背景布.svg" alt="切换结构" /></el-icon>
        <span>背景样式</span>
      </div>
      <div class="toolbar-item" @click="changeLayout">
        <el-icon><img src="@/assets/全科.svg" alt="切换结构" /></el-icon>
        <span>结构</span>
      </div>

      <el-popover :width="400" :height="100" trigger="click">
        <template #reference>
          <div class="toolbar-item">
            <el-icon
              ><img src="@/assets/mindmap主题.svg" alt="切换主题"
            /></el-icon>
            <span>主题</span>
          </div>
        </template>
        <THeme @theme-selected="changeTheme"/>
      </el-popover>

      <div class="toolbar-item" @click="defineTheme">
        <el-icon
          ><img src="@/assets/新建主题.svg" alt="注册并使用新主题"
        /></el-icon>
        <span>新建主题</span>
      </div>
    </div>

    <div class="toolbar-4">

      <div class="toolbar-item" >
        <el-icon><input type="file" @input="onInput" style="position: relative;"><img src="@/assets/导入.svg" ></el-icon>
        <span>导入</span>
      </div>

      <el-popover :width="300" :height="100" trigger="click">
        <template #reference>
      <div class="toolbar-item">
        <el-icon><img src="@/assets/导出.svg" alt="切换结构" /></el-icon>
        <span>导出</span>
      </div>
      </template>
      <div style="display:flex; border: 2px solid #333; border-radius: 10px;">
  <el-icon style="font-size:32px" @click="Exportpdf"><img src="@/assets/pdf2.svg" /></el-icon>
  <span>PDF</span>
</div>
    <div style="display:flex; border: 2px solid #333; border-radius: 10px;">
      <el-icon style="font-size:32px" @click="Exportxmind"><img src="@/assets/xmind.svg"/></el-icon>
      <span>XMIND</span>
    </div>
    <div style="display:flex; border: 2px solid #333; border-radius: 10px;">
      <el-icon style="font-size:32px" @click="Exportpng"><img src="@/assets/png.svg"></el-icon>
      <span>PNG</span>
    </div>
    <div style="display:flex; border: 2px solid #333; border-radius: 10px;">
      <el-icon style="font-size:32px" @click="Exportjson"><img src="@/assets/json.svg"/></el-icon>
      <span>JSON</span>
    </div>
    </el-popover>

      <div class="toolbar-item" @click="changeTheme">
        <el-icon><img src="@/assets/快捷键设置.svg" alt="切换主题" /></el-icon>
        <span>快捷键设置</span>
      </div>
    </div>

    <div class="toolbar-3" v-if="activeNodes.length">
      <h1>节点样式</h1>
      <div class="toolbar-item">
        <h1>文字</h1>

        <el-select
          v-model="fontFamily"
          placeholder="字体样式"
          style="width: 240px"
        >
          <el-option
            v-for="item in fontFamilyOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>

        <el-select
          v-model="fontSize"
          placeholder="字体大小"
          style="width: 120px"
        >
          <el-option
            v-for="item in fontSizeOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <div style="display: flex">
          <el-icon
            ><img src="@/assets/字体加粗.svg" @click="setbold" alt="字体"
          /></el-icon>

          <el-select
            v-model="textDecoration"
            placeholder="划线"
            style="width: 120px"
          >
            <el-option
              v-for="item in textDecorationOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </div>
        <div style="display: flex">
          <el-select
            v-model="color"
            placeholder="字体颜色"
            style="width: 120px"
          >
            <el-option
              v-for="item in colorOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-select
            v-model="lineHeight"
            placeholder="行高"
            style="width: 120px"
          >
            <el-option
              v-for="item in lineHeightOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </div>
      </div>

      <div class="toolbar-item">
        <h1>边框</h1>
        <el-select
          v-model="borderColor"
          placeholder="边框颜色"
          style="width: 240px"
        >
          <el-option
            v-for="item in options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-select
          v-model="borderDasharray"
          placeholder="边框虚线"
          style="width: 240px"
        >
          <el-option
            v-for="item in options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-select
          v-model="borderWidth"
          placeholder="边框宽度"
          style="width: 240px"
        >
          <el-option
            v-for="item in options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-select
          v-model="borderRadius"
          placeholder="边框圆角"
          style="width: 240px"
        >
          <el-option
            v-for="item in options"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-button type="primary" @click="setBorderStyle">确定</el-button>
      </div>
      <div class="toolbar-item">
        <el-icon><img src="@/assets/全科.svg" alt="设置背景" /></el-icon>
        <span>设置背景颜色</span>
        <input v-model="fillColor" placeholder="内边距Y" />
        <el-button type="primary" @click="setBg">确定</el-button>
      </div>
      <div class="toolbar-item">
        <el-icon><img src="@/assets/全科.svg" alt="设置形状" /></el-icon>
        <span>设置形状</span>
        <input v-model="shape" placeholder="形状" />
        <el-button type="primary" @click="setShape">确定</el-button>
      </div>
      <div class="toolbar-item" @click="setLineStyle">
        <el-icon><img src="@/assets/全科.svg" alt="设置线条样式" /></el-icon>
        <span>设置线条样式</span>
      </div>
      <div class="toolbar-item">
        <el-icon><img src="@/assets/全科.svg" alt="设置内边距" /></el-icon>
        <span>设置内边距</span>
        <input v-model="paddingX" type="number" placeholder="内边距X" />
        <input v-model="paddingY" type="number" placeholder="内边距Y" />
        <el-button type="primary" @click="setPadding">确定</el-button>
      </div>
    </div>

    <div
      v-if="show"
      :style="{ left: left + 'px', top: top + 'px' }"
      class="context-menu"
    >
      <div v-if="type === 'node'" style="display: flex; gap: 10px">
        <el-button type="primary" @click="copy">复制</el-button>
        <el-button type="success" @click="paste">粘贴</el-button>
        <el-button type="danger" @click="cut">剪切</el-button>
      </div>
    </div>
  </div>
</template>

<style>
.mindMapDemo {
  width: 100%;
  position: relative;
  margin-top: -30px;
  height:730px;
}

#mindMapContainer {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 730px;
  margin-top:0px;
}

#mindMapContainer * {
  margin: 0;
  padding: 0;
}

.toolbar {
  position: absolute;
  left: 10px;
  top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 10px;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  border-radius: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.toolbar-2 {
  position: absolute;
  left: 730px; /* Adjusted to be on the right side of toolbar */
  top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 10px;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  border-radius: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}
.toolbar-4 {
  position: absolute;
  left: 990px; /* Adjusted to be on the right side of toolbar */
  top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  padding: 10px;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  border-radius: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.toolbar-3 {
  position: absolute;
  left: 1220px; /* Adjusted to be on the right side of toolbar */
  top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;

  background-color: #f9f9f9;
  border: 1px solid #ddd;
  border-radius: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  padding: 10px;
}
.toolbar-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  cursor: pointer;
}

.toolbar-item el-icon img {
  width: 24px; /* 设置图标的宽度 */
  height: 24px; /* 设置图标的高度 */
}

.toolbar-item span {
  margin-top: 5px; /* 设置图标和文字之间的间距 */
}

.context-menu {
  position: absolute;
  background-color: #fff;
  border: 1px solid #ddd;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  padding: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  z-index: 1000;
  transition: all 0.3s ease;
}

.context-menu:hover {
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
  transform: translateY(-2px);
}

.context-menu .el-button {
  display: block;
  width: 100%;
  text-align: left;
}
</style>
