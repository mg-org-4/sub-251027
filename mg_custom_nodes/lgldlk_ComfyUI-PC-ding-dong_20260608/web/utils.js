import { api } from '../../scripts/api.js';
import { francMap } from './addfranc.js';
export async function request(url, method, data) {
  let formData = undefined;
  if (method === 'POST') {
    formData = new FormData();
    if (data) {
      for (const [key, value] of Object.entries(data)) {
        formData.append(key, value);
      }
    }
  } else {
    url += '?' + new URLSearchParams(data).toString();
  }

  return api.fetchApi(url, { method, body: formData });
}

const AudioContext = window.AudioContext || window.webkitAudioContext;
const audioContext = new AudioContext();
let currentSource = null;

export async function fetchAndPlayAudioSingle(filename, volume = 1) {
  // Stop any currently playing audio
  if (currentSource) {
    try {
      currentSource.stop();
      currentSource.disconnect();
    } catch (e) {}
    currentSource = null;
  }

  const response = await request(`/pc_get_audio`, 'GET', { filename });
  const audioBuffer = await audioContext.decodeAudioData(await response.arrayBuffer());
  currentSource = audioContext.createBufferSource();
  const gainNode = audioContext.createGain();
  gainNode.gain.value = volume;
  currentSource.buffer = audioBuffer;
  currentSource.connect(gainNode);
  gainNode.connect(audioContext.destination);
  currentSource.start();
  currentSource.onended = () => {
    currentSource = null;
  };
}

export async function fetchAndPlayAudio(filename, volume = 1) {
  const response = await request(`/pc_get_audio`, 'GET', { filename });
  const audioBuffer = await audioContext.decodeAudioData(await response.arrayBuffer());

  let source = audioContext.createBufferSource();
  const gainNode = audioContext.createGain();
  gainNode.gain.value = volume;

  source.buffer = audioBuffer;
  source.connect(gainNode);
  gainNode.connect(audioContext.destination);
  source.start();
  source.onended = () => {
    source = null;
  };
}

export function get_video_files() {
  return request('/pc_get_video_files', 'POST')
    .then(async (res) => (await res.json()).video_files)
    .catch((err) => {
      console.error('Error fetching video files:', err);
      return [];
    });
}

export function set_play_type(play_type) {
  return request('/pc_set_play_type', 'POST', { play_type });
}

export function play_ding_dong_text(text, pitch, rate, volume) {
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.pitch = pitch;
  utterance.rate = rate;
  utterance.volume = volume;

  if (window.pcFranc) {
    console.log('🍞 ~ play_ding_dong_text ~ francMap[window.pcFranc(text)]:', window.pcFranc(text));
    utterance.lang = francMap[window.pcFranc(text, { minLength: 3 })];
  }
  console.log('🍞 ~ play_ding_dong_text ~ utterance:', utterance);
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}
