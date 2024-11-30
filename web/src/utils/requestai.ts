import axios from 'axios'
const requestai=axios.create({
    baseURL: 'http://127.0.0.1:5000/AI',
    timeout: 50000
})
export default requestai