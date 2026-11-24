import * as ort from 'onnxruntime-node';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const modelDir = path.join(__dirname, '..', 'models', 'yamnet');

async function ensureModel() {
  fs.mkdirSync(modelDir, { recursive: true });
  const files = ['model.onnx', 'model.data', 'yamnet_class_map.csv'];
  for (const f of files) {
    const dest = path.join(modelDir, f);
    if (fs.existsSync(dest) && fs.statSync(dest).size > 100) continue;
    console.log('Downloading', f);
    const url = `https://huggingface.co/anchor-flux/yamnet-onnx/resolve/main/${f}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed ${f}: ${res.status}`);
    fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  }
}

await ensureModel();
const modelPath = path.join(modelDir, 'model.onnx');
const session = await ort.InferenceSession.create(modelPath);
console.log('Inputs:', session.inputNames);
console.log('Outputs:', session.outputNames);

const inputName = session.inputNames[0];
const len = 15600;
const data = new Float32Array(len);
const tensor = new ort.Tensor('float32', data, [len]);
const out = await session.run({ [inputName]: tensor });
for (const [k, v] of Object.entries(out)) {
  console.log(k, v.dims, v.type);
  if (v.dims.includes(521)) {
    const scores = v.data;
    let top = 0, topI = 0;
    for (let i = 0; i < 521; i++) if (scores[i] > top) { top = scores[i]; topI = i; }
    console.log('Top class index:', topI, 'score:', top);
  }
}
