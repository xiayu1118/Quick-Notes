import axios from 'axios'

const requestdb = axios.create({
    //baseURL: 'http://39.97.218.60/',
    baseURL: 'http://127.0.0.1:5000/document',
    timeout: 50000
})

export default requestdb

