import * as ort from 'onnxruntime-node';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');

// inline mel (same as yamnet-engine.js)
const SR=16000,SW=400,SH=160,FFT=512,MB=64,PF=96,PH=48;
const hann=Float32Array.from({length:SW},(_,i)=>0.5*(1-Math.cos(2*Math.PI*i/SW)));
function hzMel(h){return 2595*Math.log10(1+h/700)}
function melHz(m){return 700*(10**(m/2595)-1)}
const bins=FFT/2+1; const melM=Array.from({length:MB},()=>new Float32Array(bins));
const p=Array.from({length:MB+2},(_,i)=>melHz(hzMel(125)+(i*(hzMel(7500)-hzMel(125))/(MB+1))));
for(let m=0;m<MB;m++)for(let k=0;k<bins;k++){const hz=k*SR/FFT;if(hz<p[m]||hz>p[m+2])continue;melM[m][k]=hz<=p[m+1]?(hz-p[m])/(p[m+1]-p[m]):(p[m+2]-hz)/(p[m+2]-p[m+1]);}
function fft(re,im){const n=re.length;for(let i=1,j=0;i<n;i++){let bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];}}for(let len=2;len<=n;len<<=1){const ang=-2*Math.PI/len,wRe=Math.cos(ang),wIm=Math.sin(ang);for(let i=0;i<n;i+=len){let cRe=1,cIm=0;for(let j=0;j<len/2;j++){const uRe=re[i+j],uIm=im[i+j],vRe=re[i+j+len/2]*cRe-im[i+j+len/2]*cIm,vIm=re[i+j+len/2]*cIm+im[i+j+len/2]*cRe;re[i+j]=uRe+vRe;im[i+j]=uIm+vIm;re[i+j+len/2]=uRe-vRe;im[i+j+len/2]=uIm-vIm;const nRe=cRe*wRe-cIm*wIm;cIm=cRe*wIm+cIm*wRe;cRe=nRe;}}}}
function patches(wav){const frames=Math.floor((wav.length-SW)/SH)+1;const spec=Array.from({length:frames},()=>new Float32Array(bins));for(let f=0;f<frames;f++){const off=f*SH;const re=new Float32Array(FFT),im=new Float32Array(FFT);for(let i=0;i<SW;i++)re[i]=wav[off+i]*hann[i];fft(re,im);for(let k=0;k<bins;k++)spec[f][k]=Math.hypot(re[k],im[k]);}const lm=Array.from({length:frames},()=>new Float32Array(MB));for(let f=0;f<frames;f++)for(let m=0;m<MB;m++){let e=0;for(let k=0;k<bins;k++)e+=spec[f][k]*melM[m][k];lm[f][m]=Math.log(e+0.001);}const out=[];for(let s=0;s+PF<=frames;s+=PH){const patch=new Float32Array(PF*MB);let q=0;for(let i=0;i<PF;i++)for(let m=0;m<MB;m++)patch[q++]=lm[s+i][m];out.push(patch);}return out;}

const labels=JSON.parse(fs.readFileSync(path.join(root,'src/yamnet-labels.json'),'utf8'));
const session=await ort.InferenceSession.create(path.join(root,'models/yamnet/model.onnx'));

// 440Hz tone 1s
const wav=new Float32Array(SR);
for(let i=0;i<SR;i++)wav[i]=0.3*Math.sin(2*Math.PI*440*i/SR);
const ps=patches(wav);
console.log('patches', ps.length);
const scores=await session.run({audio:new ort.Tensor('float32',ps[0],[1,1,PF,MB])});
const raw=scores.class_scores.data;
const s=Float32Array.from(raw,x=>1/(1+Math.exp(-x)));
const top=[...s].map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,8);
for(const [v,i] of top) console.log(i, labels.classes[i].name, v.toFixed(3));
