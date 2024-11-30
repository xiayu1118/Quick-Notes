import axios from 'axios'

const request = axios.create({
    //baseURL: 'http://39.97.218.60/',
    baseURL: 'http://127.0.0.1:5000',
    timeout: 50000
})

export default request

