import request from "@/utils/request"
//用户登录-参数类型
type LoginInfo={
    user:string,
    password:string,
    code?:string
}
//用户注册-参数类型
type RegisterInfo={
    user:string,
    password:string,
    password2:string,
    code?:string
}
//注册返回类型
type RegisterResult={
    success:string
    state:number
    message:string
    content:string
}
//登录返回类型
type LoginResult={
    success:string
    state:number
    message:string
    content:string
}
//获取用户信息返回类型
type UserInfo={
   
}
export const login=(loginInfo:LoginInfo)=>{
    return request<LoginResult>({
        method:"POST",
        url:"/login",
        data:loginInfo})
}
export const register=(registerInfo:RegisterInfo)=>{
    return request<RegisterResult>({
        method:"POST",
        url:"/register",
        //data:'user=${registerInfo.phone}&password=${registerInfo.password}&password2=${registerInfo.password2}'
        data:registerInfo
    })
}

export const getInfo=()=>{
    return request<UserInfo>({
        method:"GET",
        url:"/userinfo"
    })
}

export const logout=()=>{
    return request ({
        method:"POST",
        url:"/logout",
        //data:
    })
}