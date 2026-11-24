# 听音识色 (Timbre & Color)

输入音乐，识别演奏乐器的音色。YAMNet 521 类 + Wav2Vec2 + 频谱指纹三模型融合。

## 功能

- **YAMNet 521 类**：AudioSet 全量标签，68 类乐器映射到中文音色，报告展示 Top-12 原始类别
- **Wav2Vec2**：9 类西洋乐器专用分类器
- **频谱指纹**：民乐（二胡、古筝、琵琶）及兜底识别
- **融合策略**：YAMNet 40% + Wav2Vec2 28% + 频谱 32%（西洋）；民乐以频谱为主

## 用途与 Target 判定

| 平台 | 需要? | 方式 |
|------|-------|------|
| 移动端 | 需要 | Capacitor (`android/` + `ios/`) |
| 智能硬件 | 可选 | 乐器音色识别以手机麦克风/音频文件为主，穿戴展示可选（当前搁置） |

## 技术栈

- ONNX Runtime Web（YAMNet，本地 `models/yamnet/`）
- Transformers.js（Wav2Vec2，HuggingFace 缓存）
- Web Audio API 8192 点 FFT
- Capacitor 原生包装

## 开发

```bash
npm install
npm run gen:yamnet-map   # 从 CSV 生成 src/yamnet-labels.json
npm run test:yamnet      # 验证 YAMNet mel + ONNX
npm run test:ml          # 验证 Wav2Vec2
npm run build            # 同步 → www/ android/ ios/
```

模型体积：YAMNet ~15MB（已打包）+ Wav2Vec2 ~90MB（首次联网下载）

## Harness / Loop（助手）

- 遵循根 `/Users/skat/AGENTS.md`：**Harness Engineering** + **Loop Engineering**
- Guide = 本文件；Sensor = 下方验证命令；同类失误第二次须沉淀规则或测试（棘轮）
- 开干前写清 **Done**；每切片过验证环；显式停止（成功 / 上限 / 无进展 / 人审）
- 详解：`docs/HARNESS_ENGINEERING.md`、`docs/LOOP_ENGINEERING.md`

## 验证环 / Done

```bash
npm install
npm test          # 若有
npm run build     # 或 build:web
npx cap sync android   # 若有 android/
# 可选：JAVA_HOME=…/openjdk@17 npm run apk:debug
```

**Done 示例**：测试通过（若有）+ web 产物进 `www/`/`dist/` + `cap sync` 无报错。可选穿戴未做可搁置，不纳入强制 Done。
