import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const html = path.join(root, 'index.html');
if (!fs.existsSync(html)) {
  console.error('❌ index.html missing');
  process.exit(1);
}

const publicDirs = [
  path.join(root, 'www'),
  path.join(root, 'android', 'app', 'src', 'main', 'assets', 'public'),
  path.join(root, 'ios', 'App', 'App', 'public'),
];

const copyFile = (src, dest) => {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
};

const copyDir = (srcDir, destDir) => {
  if (!fs.existsSync(srcDir)) return;
  fs.mkdirSync(destDir, { recursive: true });
  for (const name of fs.readdirSync(srcDir)) {
    const s = path.join(srcDir, name);
    const d = path.join(destDir, name);
    if (fs.statSync(s).isDirectory()) copyDir(s, d);
    else copyFile(s, d);
  }
};

for (const dir of publicDirs) {
  copyFile(html, path.join(dir, 'index.html'));
  copyDir(path.join(root, 'src'), path.join(dir, 'src'));
  copyDir(path.join(root, 'models'), path.join(dir, 'models'));
}

console.log('✅ synced index.html + src/ + models/ → www/ android/ ios/');
