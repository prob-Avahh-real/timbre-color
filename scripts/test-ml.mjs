import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = false;
env.useBrowserCache = false;

const MODEL = 'onnx-community/Musical-Instrument-Classification-ONNX';
console.log('Loading', MODEL, '...');
const classifier = await pipeline('audio-classification', MODEL);
console.log('Model loaded');

// 3s of silence at 16kHz — just verify inference runs
const samples = new Float32Array(16000 * 3);
const result = await classifier(samples, { topk: 3, sampling_rate: 16000 });
console.log('Result:', result);
