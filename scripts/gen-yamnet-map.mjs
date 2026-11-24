#!/usr/bin/env node
/** Generate src/yamnet-labels.json from models/yamnet/yamnet_class_map.csv */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const csv = fs.readFileSync(path.join(root, 'models/yamnet/yamnet_class_map.csv'), 'utf8');
const lines = csv.trim().split('\n').slice(1);

/** index → { instrument, weight } — explicit high-confidence mappings */
const EXPLICIT = {
  133: ['吉他', 0.35], 134: ['吉他', 0.5], 135: ['吉他', 1], 136: ['吉他', 0.95],
  137: ['吉他', 0.85], 138: ['吉他', 1], 139: ['吉他', 0.9], 140: ['吉他', 0.7],
  141: ['吉他', 0.85], 142: ['吉他', 0.75], 146: ['吉他', 0.8],
  143: ['琵琶', 0.7], 144: ['琵琶', 0.65],
  145: ['古筝', 0.95], 147: ['钢琴', 0.85], 148: ['钢琴', 1], 149: ['钢琴', 0.9],
  150: ['风琴', 0.85], 151: ['风琴', 0.8], 152: ['风琴', 0.85], 153: ['钢琴', 0.75],
  155: ['钢琴', 0.8],
  156: ['鼓组', 0.6], 157: ['鼓组', 1], 158: ['鼓组', 0.9], 159: ['鼓组', 0.95],
  160: ['鼓组', 0.9], 161: ['鼓组', 0.85], 162: ['鼓组', 0.85], 163: ['鼓组', 0.9],
  164: ['鼓组', 0.85], 165: ['鼓组', 0.8], 166: ['鼓组', 0.85], 167: ['鼓组', 0.9],
  168: ['鼓组', 0.7], 169: ['鼓组', 0.75], 170: ['鼓组', 0.7], 171: ['鼓组', 0.75],
  172: ['鼓组', 0.8], 173: ['鼓组', 0.75], 174: ['鼓组', 0.7], 175: ['鼓组', 0.75],
  176: ['鼓组', 0.7], 177: ['鼓组', 0.7], 178: ['鼓组', 0.75],
  180: ['小号', 0.5], 181: ['小号', 0.75], 182: ['小号', 1], 183: ['小号', 0.8],
  184: ['二胡', 0.45], 185: ['小提琴', 0.5], 186: ['小提琴', 1], 187: ['小提琴', 0.6],
  188: ['大提琴', 1], 189: ['大提琴', 0.85],
  190: ['长笛', 0.35], 191: ['长笛', 1], 192: ['萨克斯', 1], 193: ['单簧管', 1],
  194: ['古筝', 0.5], 203: ['单簧管', 0.6], 204: ['风琴', 0.8], 205: ['风琴', 0.75],
  194: ['古筝', 0.55],
};

const KEYWORD_RULES = [
  [/piano|harpsichord|keyboard|electric piano/i, '钢琴', 0.7],
  [/organ|hammond|accordion|bagpipe|harmonium/i, '风琴', 0.65],
  [/synthesizer|sampler/i, '钢琴', 0.55],
  [/violin|fiddle/i, '小提琴', 0.85],
  [/cello/i, '大提琴', 0.9],
  [/double bass|bass guitar/i, '大提琴', 0.55],
  [/bowed string/i, '二胡', 0.4],
  [/flute/i, '长笛', 0.9],
  [/saxophone/i, '萨克斯', 0.9],
  [/clarinet|oboe|woodwind|harmonica/i, '单簧管', 0.7],
  [/trumpet|trombone|french horn|brass/i, '小号', 0.75],
  [/guitar|banjo|ukulele|strum|plucked string/i, '吉他', 0.65],
  [/zither/i, '古筝', 0.85],
  [/sitar|mandolin/i, '琵琶', 0.7],
  [/harp/i, '古筝', 0.45],
  [/drum|percussion|cymbal|hi-hat|snare|tabla|timpani|marimba|xylophone|vibraphone|gong|tambourine|maraca|rimshot|drum roll|steelpan|mallet/i, '鼓组', 0.75],
  [/string section|orchestra/i, '小提琴', 0.35],
];

function mapLabel(index, name) {
  if (EXPLICIT[index]) return { instrument: EXPLICIT[index][0], weight: EXPLICIT[index][1] };
  for (const [re, inst, w] of KEYWORD_RULES) {
    if (re.test(name)) return { instrument: inst, weight: w };
  }
  return null;
}

const classes = lines.map(line => {
  const m = line.match(/^(\d+),([^,]+),"?([^"]*)"?$/);
  if (!m) {
    const parts = line.split(',');
    const index = +parts[0];
    const name = parts.slice(2).join(',').replace(/^"|"$/g, '');
    const map = mapLabel(index, name);
    return { index, mid: parts[1], name, map };
  }
  const index = +m[1];
  const name = m[3];
  return { index, mid: m[2], name, map: mapLabel(index, name) };
});

if (classes.length !== 521) console.warn('Expected 521 classes, got', classes.length);

const out = {
  version: 1,
  numClasses: classes.length,
  classes: classes.map(c => ({ index: c.index, mid: c.mid, name: c.name })),
  instrumentMap: Object.fromEntries(
    classes.filter(c => c.map).map(c => [c.index, c.map])
  ),
};

fs.writeFileSync(path.join(root, 'src/yamnet-labels.json'), JSON.stringify(out));
console.log('✅ wrote src/yamnet-labels.json —', Object.keys(out.instrumentMap).length, 'instrument mappings');
