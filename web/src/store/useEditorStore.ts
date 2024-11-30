import {ref,computed} from 'vue'
import {defineStore} from "pinia"
import { ElMessage } from 'element-plus'
interface Token{
    access_token:string
}
export const useTokenStore=defineStore('userstoken',()=>{
    const tokenJson=ref("")
    const token=computed<Token>(()=>{
        try{
            return JSON.parse(tokenJson.value||"{}")
        }
        catch(e){
            ElMessage.error("json字符串格式不对,转化对象失败")
            throw(e)
        }
    })
    function saveToken(data:string){
        tokenJson.value=data
        window.localStorage.setItem("TokenInfo",data)
    }
    return {token,saveToken}
})