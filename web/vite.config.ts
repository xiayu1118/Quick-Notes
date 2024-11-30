import {defineConfig} from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import {createSvgIconsPlugin} from 'vite-plugin-svg-icons'
import Icons from 'unplugin-icons/vite'
import IconsResolver from 'unplugin-icons/resolver'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import {ElementPlusResolver} from 'unplugin-vue-components/resolvers'
// https://vitejs.dev/config/
export default defineConfig({
    base:'/editFront1/',
    plugins: [
        vue(),
        AutoImport({
            resolvers: [
                ElementPlusResolver(),
                IconsResolver({
                    prefix: 'Icon',
                }),
            ],
        }),
        
        Components({
            resolvers: [
                ElementPlusResolver(),
                IconsResolver({
                    enabledCollections: ['ep'],
                }),
            ],
        }),
        Icons({
            autoInstall:true,
        }),
        createSvgIconsPlugin({
            // 指定需要缓存的图标文件夹
            iconDirs: [path.resolve(process.cwd(), 'src/assets/icons')],
            // 指定symbolId格式
            symbolId: 'icon-[dir]-[name]',
        })
    ],
    
    define: {
        'process.env': {}
      },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src')
            
        }
    },
    css: {
        preprocessorOptions: {
            scss: {
                javascriptEnabled: true,
                additionalData: `@use "@/styles/variables.scss" as *;`
            }
        }
    },
    server: {
        host: '0.0.0.0',
       
        proxy: {
            '/api': {
                // '/api'是代理标识，用于告诉node，url前面是/api的就是使用代理的
                ws: true,
                target: 'http://127.0.0.1:8000', //这里填入你要请求的接口的前缀
                changeOrigin: true,   //是否跨域
                rewrite: (path) => path.replace(/^\/api/, '')
            }
        }
    }
})

