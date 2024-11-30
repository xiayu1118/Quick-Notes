import {
  createRouter,
  createWebHistory,
} from "vue-router";
import { useTokenStore } from "@/store/userstoken";
const router = createRouter({
  history: createWebHistory("/editFront1/"),  // 这里指定你的仓库名称
  routes: [
    {
      path: "/dashboard",
      name: "home",
      component: () => import("@/layout/AppLayout.vue"),
      meta: { requiresAuth: true },
      children: [
        {
          path: "/dataset/:id/",
          name: "dataset",
          component: () => import("@/views/dataset.vue"),
          props:true,
        },
        {
          path:"/preview/:id/:dconame",
          name:"Preview",
          component:()=>import("@/views/preview.vue"),
          props:true
        },
        {
          path: "/Chat_rag",
          name: "chat",
          component: () => import("@/components/AI/Chat_rag.vue"),
        },
        {
          path:"/excel",
          name:"excel",
          component:()=>import("@/views/Editor/word/index.vue")
        },
        {
          path: "/doc",
          name: "doctemp",
          component: () => import("@/views/temp/Doc.vue"),
        },
        {
          path: "/ppt",
          name: "ppttemp",
          component: () => import("@/views/temp/ppt.vue"),
        },
        {
          path: "",
          component: () => import("@/views/Homepage/DocsView.vue"),
        },
        {
          path: "/wangEditor",
          name: "wangEditor",
          component: () => import("@/views/Editor/Editor.vue"),
        },
        {
          path: "/flowchart",
          name: "flowchart",
          component: () => import("@/views/temp/flowchart.vue"),
        },
        {
          path: "/editflowchart",
          name: "editflowchart",
          component: () => import("@/views/Editor/flowchart/flowchart.vue"),
        },
        {
          path:"/drawing",
          name: "mindmap",
          component: () => import("@/views/temp/draw.vue"),
        },
        {
          path: "/mindmap",
          name: "mindmaptemp",
          component: () => import("@/views/temp/mindmap.vue"),
        },
        {
          path: "/pptEditor",
          name: "pptEditor",
          component: () => import("@/views/Editor/ppt.vue"),
        },
        {
          path: "/person",
          name: "person",
          component: () => import("@/views/Person.vue"),
        },
        {
          path: "/EditPerson",
          name: "EditPerson",
          component: () => import("@/views/EditPerson.vue"),
        },
        {
          path: "/Editppt",
          name: "Editppt",
          component: () => import("@/views/Editor/ppt.vue"),
        },
        {
          path: "/Editmindmap",
          name: "Editmindmap",
          component: () => import("@/views/Editor/mindmap/mindmap.vue"),
        },
        {
          path:"/Editdraw",
          name:"Editdraw",
          component:()=>import("@/views/Editor/draw.vue"),
        },
        {
          path:"/knowledge",
          name:"knowledge",
          component:()=>import("@/views/knowledge.vue"),
        }
      ],
    },
    
    {
      path:"/test",
      name:"test",
      component:()=>import("@/layout/AppLayout.vue")
    },
    {
      path: "/",
      name: "Index",
      component: () => import("@/views/landing.vue"),
    },
    {
      path: "/setting",
      name: "setting",
      component: () => import("@/views/Setting.vue"),
    },
    {
      path: "/login",
      name: "login",
      component: () => import("@/views/login/LoginView.vue"),
    },
    {
      path: "/register",
      name: "register",
      component: () => import("@/views/signup/SignUpView.vue"),
    },
    {
      path: "/:xxx(.*)*",
      name: "ErrorPage",
      component: () => import("@/views/404.vue"),
    },
    
  ],
});

//创建路由实例

export function resetRouter() {
  router.replace({ path: "/login" });
  location.reload();
}
// 每次路由位置发生变化时要做的事：验证token

 router.beforeEach((to, _from, next) => {
       const store = useTokenStore();

       if (to.matched.some(record => record.meta.requiresAuth)) {
         console.log(store.token.access_token )
         if (!store.token.access_token ) {

             next({ name: 'login', query: { redirect: to.fullPath } });
         }
         else {next();

         }
       }

         next();

     });
export default router;
