<template>
    <div class="speech-to-text">
      <h2>语音转文字</h2>
      <div class="controls">
        <el-icon @click="startRecording" v-show="!isRecording"><VideoPlay /></el-icon>
        <el-icon @click="stopRecording" v-show="isRecording"><VideoPause /></el-icon>
        <input type="file" @change="handleFileUpload" class="file-input" />
      </div>
      <div class="progress">
        <progress :value="progress" max="100"></progress>
        <canvas ref="waveCanvas" class="wave-canvas"></canvas>
      </div>
      <div class="output">
        <h3>转录文本</h3>
        <p>{{ transcribedText }}</p>
      </div>
  
      <!-- 模态框 -->
      <div v-if="showModal" class="modal">
        <div class="modal-content">
          <span class="close" @click="closeModal">&times;</span>
          <p>没有检测到录音设备，请检查并重试。</p>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  import axios from 'axios';
  import lamejs from 'lamejs';
import  requestai  from '@/utils/requestai.ts';
  
  const transcribedText = ref('');
  const isRecording = ref(false);
  const progress = ref(0);
  const audioChunks = [];
  let mediaRecorder = null;
  let audioContext = null;
  let analyser = null;
  let dataArray = null;
  
  const showModal = ref(false);
  
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      isRecording.value = true;
      audioChunks.length = 0;
      progress.value = 0;
  
      audioContext = new AudioContext();
      analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.fftSize = 2048;
      const bufferLength = analyser.frequencyBinCount;
      dataArray = new Uint8Array(bufferLength);
  
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
      mediaRecorder.start();
  
      drawWaveform();
    } catch (err) {
      showModal.value = true;
    }
  };
  
  const stopRecording = async () => {
    isRecording.value = false;
    mediaRecorder.stop();
    audioContext.close();
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
  
      // 转码为 MP3
      const reader = new FileReader();
      reader.readAsArrayBuffer(audioBlob);
      reader.onloadend = async () => {
        const buffer = reader.result;
        const wav = lamejs.WavHeader.readHeader(new DataView(buffer));
        const samples = new Int16Array(buffer, wav.dataOffset, wav.dataLen / 2);
  
        const mp3Encoder = new lamejs.Mp3Encoder(1, wav.sampleRate, 128);
        const mp3Data = [];
        let remaining = samples.length;
        const maxSamples = 1152;
  
        for (let i = 0; remaining >= maxSamples; i += maxSamples) {
          const mono = samples.subarray(i, i + maxSamples);
          const mp3buf = mp3Encoder.encodeBuffer(mono);
          if (mp3buf.length > 0) {
            mp3Data.push(mp3buf);
          }
          remaining -= maxSamples;
        }
        const end = mp3Encoder.flush();
        if (end.length > 0) {
          mp3Data.push(end);
        }
        const mp3Blob = new Blob(mp3Data, { type: 'audio/mp3' });
        uploadAudio(mp3Blob);
      };
    };
  };
  
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    uploadAudio(file);
  };
  
  const uploadAudio = async (file) => {
    const formData = new FormData();
    formData.append('username', "admin");
    formData.append('file', file);
  
    try {
      const response = await requestai.post('/vtoc', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      transcribedText.value = response.data.text;
    } catch (error) {
      console.error('Error uploading audio:', error);
    }
  };
  
  const drawWaveform = () => {
    const canvas = document.querySelector('.wave-canvas');
    const canvasCtx = canvas.getContext('2d');
  
    const draw = () => {
      if (!isRecording.value) return;
  
      analyser.getByteTimeDomainData(dataArray);
  
      canvasCtx.fillStyle = 'rgb(200, 200, 200)';
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
  
      canvasCtx.lineWidth = 2;
      canvasCtx.strokeStyle = 'rgb(0, 0, 0)';
  
      canvasCtx.beginPath();
  
      const sliceWidth = (canvas.width * 1.0) / analyser.fftSize;
      let x = 0;
  
      for (let i = 0; i < analyser.fftSize; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
  
        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }
  
        x += sliceWidth;
      }
  
      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
  
      requestAnimationFrame(draw);
    };
  
    draw();
  };
  
  const closeModal = () => {
    showModal.value = false;
  };
  
  onMounted(() => {
    const canvas = document.querySelector('.wave-canvas');
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
  });
  </script>
  
  <style scoped>
  .speech-to-text {
    max-width: 250px;
    margin: 0 auto;
    margin-right: 100px;
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    background-color: #f9f9f9;
    font-family: Arial, sans-serif;
  }
  
  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  
  .file-input {
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
  }
  
  .progress {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
  }
  
  progress {
    width: 100%;
    height: 20px;
  }
  
  .wave-canvas {
    width: 100%;
    height: 100px;
    border: 1px solid #ccc;
    border-radius: 5px;
    margin-top: 10px;
  }
  
  .output {
    margin-top: 20px;
    background-color: #fff;
    padding: 20px;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  }
  
  .output p {
    white-space: pre-wrap;
    font-size: 16px;
  }
  
  .modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .modal-content {
    background-color: #fff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    position: relative;
    width: 80%;
    max-width: 500px;
  }
  
  .close {
    position: absolute;
    top: 10px;
    right: 10px;
    font-size: 24px;
    cursor: pointer;
  }
  </style>
  