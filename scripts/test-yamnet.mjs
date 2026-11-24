import { pipeline, env } from '@xenova/transformers';

env.allowLocalModels = false;
env.useBrowserCache = false;

const models = [
  'STMicroelectronics/yamnet',
  'thelou1s/yamnet',
  'jafet21/yamnetonnx',
  'zeropointnine/yamnet-onnx',
];

const samples = new Float32Array(16000); // 1s silence

for (const id of models) {
  try {
    console.log('\nTrying', id);
    const p = await pipeline('audio-classification', id);
    const r = await p(samples, { topk: 5, sampling_rate: 16000 });
    console.log('OK', id, r);
    break;
  } catch (e) {
    console.log('FAIL', id, e.message?.slice(0, 120));
  }
}
