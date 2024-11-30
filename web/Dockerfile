# 使用 Node.js 的官方镜像  
FROM node:16  

# 设置工作目录  
WORKDIR /app  

# 复制前端依赖文件并安装依赖包  
COPY package.json package-lock.json ./  
RUN npm install  

# 复制前端代码  
COPY . .  

# 构建前端应用  
RUN npm run build  

# 使用 NGINX 进行静态文件托管  
FROM nginx:alpine  

COPY --from=0 /app/dist /usr/share/nginx/html  

# 替换默认配置  
COPY nginx.conf /etc/nginx/conf.d/default.conf  

# 暴露端口  
EXPOSE 80