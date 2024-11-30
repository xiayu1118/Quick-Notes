import request from "@/utils/request";

// 获取所有文档
export const getalldocs = (token: string) => {
    return request({
      method: "POST",
      url: "/show_file",
      data: { user: token },
    });
  };

// 保存或新增文档
export const newdocs = (id: number) => {
  return request({
    method: "POST",
    url: '/docs/',
    data: { id }
  });
}
