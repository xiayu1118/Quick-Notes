/* empty css             *//* empty css                        *//* empty css                  *//* empty css                    *//* empty css                  */import{x as D,r as i,j as T,o as $,d as P,p as q,w as o,g as B,f as s,i as u,q as N,v as O,z as p,L as j,J as F,G as I,M,N as V,O as L,P as R}from"./index-DI7bagol.js";import{r as _}from"./requestdb-CT7yqwwj.js";import"./axios-B4uVmeYG.js";const U={style:{display:"flex"}},Y=D({__name:"dataset",setup(z){const l=i(!0),f=T(),c=i([]),n=i(f.params.id),g=async()=>{try{const e=await _.post("/show_milvus",{user:"admin",id:n.value});e.data.success==="true"&&e.data.data.length>0?c.value=e.data.data:console.log(`获取数据失败: ${e.data.message}`)}catch(e){console.log("发生错误: "+e.message)}finally{setTimeout(()=>{l.value=!1},1e3)}},h=e=>({username:"admin",ids:n.value,times:new Date().toISOString(),various:d(e.name),dconame:e.name}),w=async e=>{const t=new FormData;t.append("username","admin"),t.append("ids",n.value.toString()),t.append("times",new Date().toISOString()),t.append("various",d(e.file.name)),t.append("dconame",e.file.name),t.append("doc",e.file);try{const a=await _.post(e.action,t);a.data.success=="true"?e.onSuccess(a.data,e.file):e.onError(new Error(a.data.message))}catch(a){e.onError(a)}},v=e=>{p.info(`预览分块效果：${e}`),j.push({name:"Preview",params:{id:n.value,dconame:e}})},x=e=>{p.info(`删除文档：${e}`)};$(()=>{g()});const d=e=>{var a;switch((a=e.split(".").pop())==null?void 0:a.toLowerCase()){case"doc":case"docx":return"Word 文档";case"ppt":case"pptx":return"PPT 文件";case"xls":case"xlsx":return"Excel 文件";case"pdf":return"PDF 文件";case"jpg":return"jpg 文件";default:return"未知文件类型"}},y=(e,t)=>{if(e.success){p.success("上传成功");const a=d(t.name);c.value.push({date:new Date().toISOString().split("T")[0],name:t.name,sorts:a})}else console.log(`上传失败: ${e.message}`)};return(e,t)=>{const a=F,E=I,r=M,b=V,S=L,C=R;return P(),q(C,{style:{width:"auto"}},{header:o(()=>[B("div",U,[s(E,{action:"/create_words","on-success":y,data:h,"show-file-list":!1,"http-request":w},{default:o(()=>[s(a,{type:"primary"},{default:o(()=>[u("上传")]),_:1})]),_:1})])]),default:o(()=>[N(s(b,{data:c.value,border:"",style:{width:"auto",display:"flex"}},{default:o(()=>[s(r,{prop:"date",label:"修改日期",width:"180",align:"center"}),s(r,{prop:"name",label:"名称",width:"400",align:"center"}),s(r,{prop:"sorts",label:"类型",align:"center"}),s(r,{prop:"address",label:"操作",align:"center"},{default:o(({row:m})=>[s(a,{onClick:k=>v(m.name)},{default:o(()=>[u("预览分块效果")]),_:2},1032,["onClick"]),s(a,{onClick:k=>x(m.name)},{default:o(()=>[u("删除")]),_:2},1032,["onClick"])]),_:1})]),_:1},8,["data"]),[[O,!l.value]]),s(S,{loading:l.value,rows:5,animated:""},null,8,["loading"])]),_:1})}}});export{Y as default};