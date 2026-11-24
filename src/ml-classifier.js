import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
import {
  preloadYamnet, isYamnetReady, classifyYamnetBuffer, classifyYamnetLive,
  YAMNET_CLASSES,
} from './yamnet-engine.js';

env.allowLocalModels = false;
env.useBrowserCache = true;

const WAV2VEC_MODEL = 'onnx-community/Musical-Instrument-Classification-ONNX';
const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = SAMPLE_RATE * 3;

export { YAMNET_CLASSES };

export const ML_LABEL_TO_NAME = {
  Acoustic_Guitar: '吉他',
  Bass_Guitar: '吉他',
  Drum_set: '鼓组',
  Electro_Guitar: '吉他',
  flute: '长笛',
  Hi_Hats: '鼓组',
  Keyboard: '钢琴',
  Trumpet: '小号',
  Violin: '小提琴',
};

let wav2vec = null;
let loadPromise = null;
let loadPct = 0;
let loadStage = 'idle';

export function isReady() { return !!wav2vec; }
export function isYamnetLoaded() { return isYamnetReady(); }
export function getLoadProgress() { return loadPct; }
export function getLoadStage() { return loadStage; }

export function preload(onProgress) {
  if (!loadPromise) {
    loadPromise = init(onProgress).catch(err => {
      loadPromise = null;
      throw err;
    });
  }
  return loadPromise;
}

async function init(onProgress) {
  loadStage = 'yamnet-labels';
  loadPct = 0;
  onProgress?.(0, 'YAMNet');

  await preloadYamnet((pct, tag) => {
    loadPct = Math.round(pct * 0.45);
    loadStage = tag === 'model' ? 'yamnet-model' : 'yamnet';
    onProgress?.(loadPct, loadStage);
  });

  loadStage = 'wav2vec2';
  wav2vec = await pipeline('audio-classification', WAV2VEC_MODEL, {
    progress_callback: data => {
      if (data.status === 'progress' && data.progress != null) {
        loadPct = 45 + Math.round(data.progress * 0.55);
        onProgress?.(loadPct, 'wav2vec2');
      }
    },
  });

  loadPct = 100;
  loadStage = 'done';
  onProgress?.(100, 'done');
  return { wav2vec, yamnet: true };
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    out[i] = (input[idx] || 0) + ((input[idx + 1] || 0) - (input[idx] || 0)) * frac;
  }
  return out;
}

function mixToMono(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) return audioBuffer.getChannelData(0).slice();
  const len = audioBuffer.length;
  const out = new Float32Array(len);
  const n = audioBuffer.numberOfChannels;
  for (let c = 0; c < n; c++) {
    const ch = audioBuffer.getChannelData(c);
    for (let i = 0; i < len; i++) out[i] += ch[i] / n;
  }
  return out;
}

function chunkRms(chunk) {
  let s = 0;
  for (let i = 0; i < chunk.length; i++) s += chunk[i] * chunk[i];
  return Math.sqrt(s / chunk.length);
}

async function classifyWav2VecBuffer(audioBuffer, { hopSec = 1.5, onProgress } = {}) {
  if (!wav2vec) await preload();
  const mono = resample(mixToMono(audioBuffer), audioBuffer.sampleRate, SAMPLE_RATE);
  const hop = Math.floor(SAMPLE_RATE * hopSec);
  const totals = {};
  const segments = [];
  let windows = 0;
  for (let start = 0; start + CHUNK_SAMPLES <= mono.length; start += hop) windows++;

  let w = 0;
  for (let start = 0; start + CHUNK_SAMPLES <= mono.length; start += hop, w++) {
    const chunk = mono.subarray(start, start + CHUNK_SAMPLES);
    if (chunkRms(chunk) < 0.008) continue;
    const preds = await wav2vec(chunk, { topk: 3, sampling_rate: SAMPLE_RATE });
    for (const p of preds) {
      const name = ML_LABEL_TO_NAME[p.label];
      if (name) totals[name] = (totals[name] || 0) + p.score;
    }
    const topName = ML_LABEL_TO_NAME[preds[0]?.label];
    if (topName) segments.push({ t: start / SAMPLE_RATE, name: topName, score: preds[0].score });
    if (onProgress && windows > 0) onProgress((w / windows) * 100);
    if (w % 2 === 1) await new Promise(r => setTimeout(r, 0));
  }

  const sum = Object.values(totals).reduce((a, b) => a + b, 0) || 1;
  return {
    ranked: Object.entries(totals).map(([name, v]) => ({ name, pct: v / sum })).sort((a, b) => b.pct - a.pct),
    segments,
    totals,
  };
}

/** 三模型离线分析 */
export async function classifyAudioBuffer(audioBuffer, opts = {}) {
  const { onProgress, onStage } = opts;
  if (!wav2vec) await preload();

  onStage?.('yamnet');
  const yam = await classifyYamnetBuffer(audioBuffer, {
    onProgress: p => onProgress?.(p * 0.55, 'yamnet'),
  });

  onStage?.('wav2vec2');
  const w2v = await classifyWav2VecBuffer(audioBuffer, {
    hopSec: 1.5,
    onProgress: p => onProgress?.(55 + p * 0.45, 'wav2vec2'),
  });

  return {
    yamnet: yam,
    wav2vec: w2v,
    ranked: yam.ranked,
    segments: yam.segments.length ? yam.segments : w2v.segments,
    totals: yam.totals,
  };
}

/** 实时融合推理 */
export async function classifyLiveChunk(samples, sampleRate) {
  const [yam, w2v] = await Promise.all([
    classifyYamnetLive(samples, sampleRate),
    (async () => {
      if (!wav2vec) return null;
      const mono = resample(samples, sampleRate, SAMPLE_RATE);
      let chunk;
      if (mono.length >= CHUNK_SAMPLES) chunk = mono.subarray(mono.length - CHUNK_SAMPLES);
      else {
        chunk = new Float32Array(CHUNK_SAMPLES);
        chunk.set(mono, CHUNK_SAMPLES - mono.length);
      }
      if (chunkRms(chunk) < 0.008) return null;
      return wav2vec(chunk, { topk: 3, sampling_rate: SAMPLE_RATE });
    })(),
  ]);
  return { yamnet: yam, wav2vec: w2v };
}
