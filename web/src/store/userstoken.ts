import { ref, computed } from 'vue'
import { defineStore } from "pinia"
import { ElMessage } from 'element-plus'

interface Token {
    access_token: string
}

const TOKEN_EXPIRATION_TIME = 60 * 60 * 1000; // 60 minutes in milliseconds

export const useTokenStore = defineStore('userstoken', () => {
    const tokenJson = ref(window.localStorage.getItem("TokenInfo") || "")
    const tokenTimestamp = ref(Number(window.localStorage.getItem("TokenTimestamp")) || 0)

    const token = computed<Token>(() => {
        try {
            const currentTime = new Date().getTime();
            if (currentTime - tokenTimestamp.value > TOKEN_EXPIRATION_TIME) {
                // Token expired
                ElMessage.error("Token 已过期，请重新登录")
                clearToken()
                return {} as Token;
            }
            return JSON.parse(tokenJson.value || "{}")
        } catch (e) {
            ElMessage.error("json字符串格式不对,转化对象失败")
            throw (e)
        }
    })

    function saveToken(data: string) {
        tokenJson.value = data
        tokenTimestamp.value = new Date().getTime()
        window.localStorage.setItem("TokenInfo", data)
        window.localStorage.setItem("TokenTimestamp", tokenTimestamp.value.toString())
    }

    function clearToken() {
        tokenJson.value = ""
        tokenTimestamp.value = 0
        window.localStorage.removeItem("TokenInfo")
        window.localStorage.removeItem("TokenTimestamp")
    }

    return { token, saveToken, clearToken }
})

export const useUserAvatarStore = defineStore('userAvatar', {
  state: () => ({
    avatarUrl: 'https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png',
  }),
  actions: {
    setAvatarUrl(newUrl: string) {
      this.avatarUrl = newUrl;
    },
  },
});
