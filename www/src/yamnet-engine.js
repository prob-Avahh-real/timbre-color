/**
 * YAMNet ONNX inference — mel patch frontend + 521-class AudioSet output.
 * Model: anchor-flux/yamnet-onnx (input shape [1,1,96,64] log-mel patch)
 */
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.mjs';

const SAMPLE_RATE = 16000;
const STFT_WINDOW = 400;
const STFT_HOP = 160;
const FFT_SIZE = 512;
const MEL_BANDS = 64;
const MEL_MIN = 125;
const MEL_MAX = 7500;
const LOG_OFFSET = 0.001;
const PATCH_FRAMES = 96;
const PATCH_HOP_FRAMES = 48;
const MODEL_BASE = './models/yamnet/';

let session = null;
let melMatrix = null;
let hann = null;
let loadPromise = null;
let labelsData = null;

export let YAMNET_CLASSES = [];
export let YAMNET_INSTRUMENT_MAP = {};

export function isYamnetReady() { return !!session; }

export function preloadYamnet(onProgress) {
  if (!loadPromise) {
    loadPromise = init(onProgress).catch(e => { loadPromise = null; throw e; });
  }
  return loadPromise;
}

async function init(onProgress) {
  onProgress?.(0, 'init');
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
  const res = await fetch('./src/yamnet-labels.json');
  labelsData = await res.json();
  YAMNET_CLASSES = labelsData.classes;
  YAMNET_INSTRUMENT_MAP = labelsData.instrumentMap;
  melMatrix = buildMelMatrix();
  hann = buildHann(STFT_WINDOW);
  onProgress?.(30, 'model');
  session = await ort.InferenceSession.create(`${MODEL_BASE}model.onnx`, {
    executionProviders: ['wasm'],
  });
  onProgress?.(100, 'done');
  return session;
}

function buildHann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / n));
  return w;
}

function hzToMel(hz) {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel) {
  return 700 * (10 ** (mel / 2595) - 1);
}

function buildMelMatrix() {
  const bins = FFT_SIZE / 2 + 1;
  const matrix = Array.from({ length: MEL_BANDS }, () => new Float32Array(bins));
  const melMin = hzToMel(MEL_MIN);
  const melMax = hzToMel(MEL_MAX);
  const points = Array.from({ length: MEL_BANDS + 2 }, (_, i) =>
    melToHz(melMin + (i * (melMax - melMin)) / (MEL_BANDS + 1))
  );
  for (let m = 0; m < MEL_BANDS; m++) {
    const left = points[m];
    const center = points[m + 1];
    const right = points[m + 2];
    for (let k = 0; k < bins; k++) {
      const hz = (k * SAMPLE_RATE) / FFT_SIZE;
      if (hz < left || hz > right) continue;
      matrix[m][k] = hz <= center
        ? (hz - left) / (center - left)
        : (right - hz) / (right - center);
    }
  }
  return matrix;
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

function stftMag(waveform) {
  const frames = Math.floor((waveform.length - STFT_WINDOW) / STFT_HOP) + 1;
  if (frames <= 0) return [];
  const bins = FFT_SIZE / 2 + 1;
  const spec = Array.from({ length: frames }, () => new Float32Array(bins));
  for (let f = 0; f < frames; f++) {
    const off = f * STFT_HOP;
    const re = new Float32Array(FFT_SIZE);
    const im = new Float32Array(FFT_SIZE);
    for (let i = 0; i < STFT_WINDOW; i++) {
      re[i] = (waveform[off + i] || 0) * hann[i];
    }
    fft(re, im);
    for (let k = 0; k < bins; k++) spec[f][k] = Math.hypot(re[k], im[k]);
  }
  return spec;
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len / 2] * curRe - im[i + j + len / 2] * curIm;
        const vIm = re[i + j + len / 2] * curIm + im[i + j + len / 2] * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe;
        im[i + j + len / 2] = uIm - vIm;
        const nRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nRe;
      }
    }
  }
}

function waveformToPatches(waveform) {
  const spec = stftMag(waveform);
  if (!spec.length) return [];
  const frames = spec.length;
  const logMel = Array.from({ length: frames }, () => new Float32Array(MEL_BANDS));
  for (let f = 0; f < frames; f++) {
    for (let m = 0; m < MEL_BANDS; m++) {
      let e = 0;
      for (let k = 0; k < spec[f].length; k++) e += spec[f][k] * melMatrix[m][k];
      logMel[f][m] = Math.log(e + LOG_OFFSET);
    }
  }
  const patches = [];
  for (let start = 0; start + PATCH_FRAMES <= frames; start += PATCH_HOP_FRAMES) {
    const patch = new Float32Array(PATCH_FRAMES * MEL_BANDS);
    let p = 0;
    for (let i = 0; i < PATCH_FRAMES; i++) {
      for (let m = 0; m < MEL_BANDS; m++) patch[p++] = logMel[start + i][m];
    }
    patches.push({ data: patch, t: (start * STFT_HOP) / SAMPLE_RATE });
  }
  return patches;
}

async function runPatch(patchData) {
  const input = new ort.Tensor('float32', patchData, [1, 1, PATCH_FRAMES, MEL_BANDS]);
  const out = await session.run({ audio: input });
  const raw = out.class_scores.data;
  const scores = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) scores[i] = 1 / (1 + Math.exp(-raw[i]));
  return scores;
}

function aggregateInstrumentScores(classScores) {
  const inst = {};
  for (let i = 0; i < classScores.length; i++) {
    const score = classScores[i];
    if (score < 0.01) continue;
    const map = YAMNET_INSTRUMENT_MAP[i] || YAMNET_INSTRUMENT_MAP[String(i)];
    if (!map) continue;
    const { instrument, weight } = map;
    inst[instrument] = (inst[instrument] || 0) + score * weight;
  }
  return inst;
}

function topRawClasses(classScores, topk = 8) {
  const arr = [];
  for (let i = 0; i < classScores.length; i++) {
    if (classScores[i] > 0.02) arr.push({ index: i, name: YAMNET_CLASSES[i].name, score: classScores[i] });
  }
  return arr.sort((a, b) => b.score - a.score).slice(0, topk);
}

function mergeScoreMaps(maps) {
  const out = {};
  for (const m of maps) {
    for (const [k, v] of Object.entries(m)) out[k] = (out[k] || 0) + v;
  }
  return out;
}

function toRanked(totals) {
  const sum = Object.values(totals).reduce((a, b) => a + b, 0) || 1;
  return Object.entries(totals)
    .map(([name, v]) => ({ name, pct: v / sum }))
    .sort((a, b) => b.pct - a.pct);
}

function chunkRms(waveform) {
  let s = 0;
  for (let i = 0; i < waveform.length; i++) s += waveform[i] * waveform[i];
  return Math.sqrt(s / waveform.length);
}

/** 全曲 YAMNet 521 类分析 */
export async function classifyYamnetBuffer(audioBuffer, { onProgress } = {}) {
  if (!session) await preloadYamnet();
  const mono = resample(mixToMono(audioBuffer), audioBuffer.sampleRate, SAMPLE_RATE);
  const patches = waveformToPatches(mono);
  const instTotals = {};
  const rawTotals = new Float32Array(521);
  const segments = [];
  let done = 0;

  for (const patch of patches) {
    const patchStart = Math.floor(patch.t * SAMPLE_RATE);
    const patchEnd = patchStart + Math.floor(PATCH_FRAMES * STFT_HOP);
    const slice = mono.subarray(patchStart, Math.min(patchEnd, mono.length));
    if (slice.length > 100 && chunkRms(slice) < 0.008) { done++; continue; }

    const scores = await runPatch(patch.data);
    for (let i = 0; i < 521; i++) rawTotals[i] += scores[i];

    const inst = aggregateInstrumentScores(scores);
    for (const [name, v] of Object.entries(inst)) instTotals[name] = (instTotals[name] || 0) + v;

    let bestI = 0, bestV = 0;
    for (let i = 0; i < 521; i++) if (scores[i] > bestV) { bestV = scores[i]; bestI = i; }
    const map = YAMNET_INSTRUMENT_MAP[bestI] || YAMNET_INSTRUMENT_MAP[String(bestI)];
    if (map && bestV > 0.15) {
      segments.push({ t: patch.t, name: map.instrument, score: bestV, rawName: YAMNET_CLASSES[bestI].name });
    }

    done++;
    if (onProgress && patches.length) onProgress((done / patches.length) * 100);
    if (done % 3 === 0) await new Promise(r => setTimeout(r, 0));
  }

  return {
    ranked: toRanked(instTotals),
    segments,
    totals: instTotals,
    rawTop: topRawClasses(rawTotals),
    allClasses: YAMNET_CLASSES.map((c, i) => ({ ...c, score: rawTotals[i] }))
      .filter(c => c.score > 0.05)
      .sort((a, b) => b.score - a.score),
  };
}

/** 实时：最近 ~1.5s 音频 */
export async function classifyYamnetLive(samples, sampleRate) {
  if (!session) return null;
  const mono = resample(samples, sampleRate, SAMPLE_RATE);
  const need = STFT_HOP * (PATCH_FRAMES - 1) + STFT_WINDOW;
  let wav = mono;
  if (wav.length < need) {
    const padded = new Float32Array(need);
    padded.set(wav, need - wav.length);
    wav = padded;
  } else if (wav.length > need * 2) {
    wav = wav.subarray(wav.length - need * 2);
  }
  if (chunkRms(wav) < 0.008) return null;

  const patches = waveformToPatches(wav);
  if (!patches.length) return null;
  const last = patches[patches.length - 1];
  const scores = await runPatch(last.data);
  return {
    instruments: toRanked(aggregateInstrumentScores(scores)),
    rawTop: topRawClasses(scores, 5),
  };
}

export { SAMPLE_RATE as YAMNET_SAMPLE_RATE };
