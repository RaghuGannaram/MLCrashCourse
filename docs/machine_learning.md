# 🧭 AI Model Formats & Runtimes — **Cheat Sheet**

_A one‑file, copy‑pasteable reference for picking the right **model format**, **runtime**, **hub/registry**, and **deployment path** for modern AI systems (LLMs, ASR/TTS, CV, multimodal). Optimized for local-first builders._

---

## Table of Contents

-   [1. Model Formats — What, Why, Pros/Cons](#1-model-formats--what-why-proscons)
    -   [1.1 Training / Checkpoint Formats](#11-training--checkpoint-formats)
    -   [1.2 Exchange / Deployment Formats](#12-exchange--deployment-formats)
    -   [1.3 Compiled / Engine / Quantized Formats](#13-compiled--engine--quantized-formats)
    -   [1.4 Convertibility & Reversibility Cheatsheet](#14-convertibility--reversibility-cheatsheet)
-   [2. Runtimes (Engines) — Who Executes the Model](#2-runtimes-engines--who-executes-the-model)
-   [3. Registries / Model Hubs](#3-registries--model-hubs)
-   [4. Compatibility Matrix (Task → Format → Runtime)](#4-compatibility-matrix-task--format--runtime)
-   [5. Decision Flows (Mermaid Diagrams)](#5-decision-flows-mermaid-diagrams)
-   [6. Practical How‑Tos & One‑Liners](#6-practical-how-tos--one-liners)
-   [7. Best Practices & Pitfalls](#7-best-practices--pitfalls)
-   [8. Glossary](#8-glossary)
-   [9. Quick Profiles (Copy‑ready Stacks)](#9-quick-profiles-copy-ready-stacks)
-   [10. Licensing & Safety Notes](#10-licensing--safety-notes)

---

## 1. Model Formats — What, Why, Pros/Cons

### 1.1 Training / Checkpoint Formats

> **Editable**; great for training/fine‑tuning. Usually **not** the final deploy format.

| Format                 | Extensions              | Category                   | Typical Producers        | Typical Consumers     | Strengths                                      | Watch‑outs                                     | Reversible\*              | Common Conversions                                            |
| ---------------------- | ----------------------- | -------------------------- | ------------------------ | --------------------- | ---------------------------------------------- | ---------------------------------------------- | ------------------------- | ------------------------------------------------------------- |
| **PyTorch Checkpoint** | `.pt`, `.pth`, `.bin`   | Weights (framework‑native) | PyTorch                  | PyTorch / LibTorch    | Full fidelity; fine‑tune ready; huge ecosystem | Python dep; version coupling; bigger footprint | ✔ to/from **safetensors** | → ONNX, TorchScript, TensorRT (via ONNX), Core ML (via tools) |
| **safetensors**        | `.safetensors`          | Weights container          | PyTorch, HF Transformers | PyTorch (safe loader) | **Safe** (no pickle), zero‑copy, fast load     | Needs model code; not a graph                  | ✔ with PyTorch weights    | Same as PyTorch once loaded                                   |
| **TensorFlow**         | SavedModel dir, `.ckpt` | Weights + graph            | TensorFlow/Keras         | TF Serving            | First‑class in TF stack                        | TF‑only tooling; versioning                    | ✔ within TF               | → TFLite, sometimes → ONNX                                    |
| **Flax/JAX**           | various                 | Weights                    | JAX/Flax                 | JAX/Flax              | Fast research & TPU                            | Smaller serving ecosystem                      | ✔ within JAX              | → ONNX (via export), → PyTorch (adapters)                     |

\* _Reversible = you can reliably get back to usable weights in the original framework._

---

### 1.2 Exchange / Deployment Formats

> **Portable graphs**; optimized for inference and conversion across platforms.

| Format          | Extensions    | Category        | Strengths                                                              | Watch‑outs                                       | Good For                   | Runtime(s)                                                                      |
| --------------- | ------------- | --------------- | ---------------------------------------------------------------------- | ------------------------------------------------ | -------------------------- | ------------------------------------------------------------------------------- |
| **ONNX**        | `.onnx`       | Graph + weights | Cross‑framework; broad tooling; ORT supports CPU/GPU/DirectML/OpenVINO | Export quirks; op coverage varies; not trainable | CV/ASR/TTS/multimodal      | **ONNX Runtime**, TensorRT (via convert), TFLite/Core ML/OpenVINO (via convert) |
| **TorchScript** | `.pt`         | Graph (PyTorch) | Python‑free serving via LibTorch                                       | Version coupling; tracing pitfalls               | C++ deployments w/ PyTorch | LibTorch                                                                        |
| **TFLite**      | `.tflite`     | Mobile graph    | Tiny, INT8‑friendly, XNNPack accel                                     | Limited op set; mobile focus                     | Android/edge               | TFLite Interpreter                                                              |
| **Core ML**     | `.mlmodel`    | Apple graph     | Apple Silicon accel; iOS/macOS                                         | Apple‑only toolchain                             | iOS/macOS                  | Core ML runtime                                                                 |
| **OpenVINO IR** | `.xml/.bin`   | Intel IR        | Great CPU/iGPU perf; INT8                                              | Intel‑centric                                    | Intel desktops/edge        | OpenVINO runtime                                                                |
| **NCNN**        | `.param/.bin` | Mobile graph    | Lean on Android (CPU/Vulkan)                                           | Niche tooling                                    | Android                    | NCNN runtime                                                                    |
| **MNN**         | `.mnn`        | Mobile graph    | Lightweight, edge‑friendly                                             | Smaller community                                | Mobile/edge                | MNN runtime                                                                     |

---

### 1.3 Compiled / Engine / Quantized Formats

> **Fastest** to run, **least flexible**. Treat as **one‑way** targets.

| Format                 | Extensions | Category              | Strengths                                    | Watch‑outs                                       | Good For                  | Runtime(s)                               |
| ---------------------- | ---------- | --------------------- | -------------------------------------------- | ------------------------------------------------ | ------------------------- | ---------------------------------------- |
| **TensorRT Engine**    | `.plan`    | GPU engine            | Top NVIDIA latency/throughput; FP16/INT8     | Tied to GPU arch/driver; rebuild on change       | Production NVIDIA serving | TensorRT                                 |
| **GGUF** (LLMs)        | `.gguf`    | Quantized LLM weights | Tiny RAM/VRAM; runs on CPU/Metal/CUDA/Vulkan | LLM‑only; **not reversible**; training info gone | Local LLM chat            | **llama.cpp** family (Ollama, LM Studio) |
| **IREE/TVM artifacts** | various    | Compiled kernels      | Great on edge/heterogeneous targets          | Setup complexity                                 | Edge accelerators         | IREE/TVM runtimes                        |

---

### 1.4 Convertibility & Reversibility Cheatsheet

```
PyTorch ↔ safetensors           # lossless
PyTorch → ONNX                  # common deploy path
ONNX → TensorRT (engine)        # best NVIDIA perf
ONNX → TFLite/Core ML/OpenVINO  # via converters (ops may need fixes)
HF LLM (PyTorch) → GGUF         # via llama.cpp tools (one-way)
TensorRT/TFLite/Core ML → ONNX  # generally NO (treat as terminal)
Quantized (INT4/INT8/GGUF) → FP # NO (information lost)
```

> **Rule of thumb:** keep a **trainable source** (PyTorch/safetensors or TF). Export **ONNX** for portability; compile to **engines** for max perf. Use **GGUF** for LLMs in local apps.

---

## 2. Runtimes (Engines) — Who Executes the Model

| Runtime                                       | Runs Formats                 | Hardware Backends                                           | When to Use                                       | Notes                                                       |
| --------------------------------------------- | ---------------------------- | ----------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------- |
| **ONNX Runtime (ORT)**                        | ONNX                         | CPU, CUDA, TensorRT, DirectML (Win), OpenVINO, CoreML (EPs) | Default for CV/ASR/TTS/multimodal                 | Graph fusions, quant, breadth; great general‑purpose engine |
| **llama.cpp** (via **Ollama**, **LM Studio**) | GGUF (LLMs)                  | CPU, Metal, CUDA, Vulkan                                    | Local LLM chat/agents                             | Streaming tokens; minimal RAM; huge ecosystem               |
| **TensorRT**                                  | ONNX → Engine                | NVIDIA GPUs                                                 | Lowest latency prod serving                       | Build step; pin to driver/arch                              |
| **TFLite**                                    | TFLite                       | CPU, GPU, Edge‑TPUs                                         | Mobile/embedded                                   | Tiny, XNNPack, NNAPI                                        |
| **Core ML**                                   | MLModel                      | Apple Silicon                                               | iOS/macOS apps                                    | Accelerated on Apple devices                                |
| **OpenVINO**                                  | IR                           | Intel CPU/iGPU/VPU                                          | Intel edge/desktop                                | Strong INT8 perf                                            |
| **NCNN / MNN**                                | NCNN/MNN                     | Android CPUs/GPUs                                           | Android‑first                                     | Lean/mobile friendly                                        |
| **PyTorch / TF**                              | Native ckpts                 | CPU/CUDA                                                    | Research, fine‑tune, or serving if stack is fixed | Heavier runtime; max flexibility                            |
| **Web runtimes**                              | ONNX (ORT‑web), WebNN/WebGPU | Browser                                                     | In‑browser demos                                  | No install; perf varies                                     |

---

## 3. Registries / Model Hubs

| Hub                    | Scope             | Common Formats                          | How to Pull                      | Notes                                  |
| ---------------------- | ----------------- | --------------------------------------- | -------------------------------- | -------------------------------------- |
| **Hugging Face Hub**   | All tasks         | PyTorch/safetensors, ONNX, GGUF, TFLite | `git lfs`, `hf_hub_download`, UI | De‑facto model registry & ecosystem    |
| **Ollama Library**     | LLMs (GGUF)       | GGUF                                    | `ollama pull model`              | Easy local LLMs; recipes (`Modelfile`) |
| **LM Studio Gallery**  | LLMs              | GGUF (pulls from HF)                    | GUI                              | Friendly desktop client                |
| **ONNX Model Zoo**     | ONNX examples     | ONNX                                    | Git/zip                          | Good starters for ORT                  |
| **TensorFlow Hub**     | TF models         | SavedModel                              | `tensorflow_hub`                 | TF‑centric                             |
| **NVIDIA NGC**         | NVIDIA‑opt models | ONNX/TensorRT                           | `ngc`/UI                         | Optimized for NVIDIA stack             |
| **OpenVINO Model Zoo** | Intel‑opt models  | OpenVINO IR                             | `omz_downloader`                 | Intel edge                             |
| **ModelScope**         | Broad             | Many                                    | CLI/UI                           | Popular esp. in APAC                   |

---

## 4. Compatibility Matrix (Task → Format → Runtime)

| Task                        | **Best Format**                     | **Best Runtime**                       | Alt / Notes                                                                   |
| --------------------------- | ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| **LLM (chat/agents)**       | **GGUF** (quant)                    | **llama.cpp** via **Ollama/LM Studio** | For max NVIDIA perf: TensorRT‑LLM (complex). Keep PyTorch ckpt for fine‑tune. |
| **ASR (streaming)**         | **ONNX** (streaming transducer/CTC) | **sherpa‑onnx** (uses ORT)             | Built‑in endpointing; CPU‑friendly (int8).                                    |
| **TTS**                     | **ONNX** voice                      | **Piper** (ORT under hood)             | Small, fast, local voices.                                                    |
| **Vision (CV)**             | **ONNX**                            | **ONNX Runtime**                       | Convert to TFLite/Core ML for mobile; to TensorRT for servers.                |
| **Multimodal (CLIP, etc.)** | **ONNX**                            | **ONNX Runtime**                       | Check op coverage; sometimes PyTorch is simpler.                              |

---

## 5. Decision Flows (Mermaid Diagrams)

```sql
                        Task?
                        └─ Is it an LLM?
                        ├─ YES  → Use **GGUF**
                        │          └─ Runtime: **llama.cpp** (via **Ollama** / **LM Studio**)
                        └─ NO   → Export/Use **ONNX**
                                    └─ Target?
                                        ├─ NVIDIA Server   → **TensorRT Engine**
                                        ├─ Mobile          → **TFLite** / **Core ML**
                                        ├─ Intel Edge      → **OpenVINO IR**
                                        └─ Desktop/General → **ONNX Runtime**

```

```sql
                                                    +-------------------+
                                                    |  ONNX Runtime     |
                                                    |  (serve ONNX)     |
                                                    +---------^---------+
                                                                |
                                                                |
                    +---------------------------+      export    |
                    | Source weights (trainable)|  --------------+
                    |  • PyTorch / TF           |                 \
                    +-------------+-------------+                  \
                                |                                 \
                                | convert (LLMs)                   \
                                v                                   \
                            +----------+                              v
                            |  GGUF    |                         +---------+
                            | (LLMs)   |                         |  ONNX   |
                            +-----+----+                         +----+----+
                                |                                   |
                                | serve                              | compile
                                v                                   v
                        +---------------------+                +-------------+
                        | llama.cpp runtimes  |                | TensorRT    |
                        | (Ollama / LM Studio)|                | Engine (.plan)
                        +---------------------+                +-------------+

                    From ONNX (besides TensorRT), you can also convert to:

                    • TFLite  (mobile/embedded)
                    • Core ML (Apple/iOS/macOS)

                    (Keep ONNX as the portable “source of truth” for deployments.)

```

---

## 6. Practical How‑Tos & One‑Liners

### 6.1 PyTorch ➜ ONNX (opset 17, dynamic batch)

```python
import torch, torch.onnx as onnx
model.eval()
dummy = torch.randn(1, 3, 224, 224)
onnx.export(
  model, dummy, "model.onnx",
  input_names=["input"], output_names=["logits"],
  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
  opset_version=17, do_constant_folding=True,
)
```

### 6.2 Validate ONNX with ONNX Runtime

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"input": np.zeros((1,3,224,224), np.float32)})
```

### 6.3 Build TensorRT Engine (example CLI)

```bash
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 --workspace=4096
```

### 6.4 TF SavedModel ➜ TFLite (INT8)

```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite = converter.convert()
open("model.tflite","wb").write(tflite)
```

### 6.5 Core ML (from ONNX)

```python
import coremltools as ct
mlmodel = ct.converters.onnx.convert(model="model.onnx")
mlmodel.save("model.mlmodel")
```

### 6.6 OpenVINO (from ONNX)

```bash
mo --input_model model.onnx --output_dir ir_out --compress_to_fp16
```

### 6.7 NCNN (from ONNX)

```bash
onnx2ncnn model.onnx model.param model.bin
```

### 6.8 LLM to GGUF (conceptual)

-   Prefer **prebuilt GGUF** on HF/Ollama.
-   For LLaMA‑family: use **llama.cpp** conversion scripts (`convert.py`) to produce `.gguf`, then quantize with `quantize` to Q4/Q5 variants.

### 6.9 Run LLM locally (Ollama)

```bash
ollama pull llama3.2:3b-instruct
curl http://localhost:11434/api/generate -d '{"model":"llama3.2:3b-instruct","prompt":"Hello!"}'
```

### 6.10 Streaming ASR (sherpa‑onnx, mic quick check)

```python
# see sherpa_quickcheck.py pattern:
# - load tokens/encoder/decoder/joiner
# - create OnlineRecognizer with endpoint rules
# - feed 20ms frames; read partials; on rec.is_endpoint(stream) -> final
```

### 6.11 Piper TTS (CLI)

```bash
echo "Hello from Piper" | piper -m voices/en_US-amy-low.onnx --output_file out.wav && (play out.wav || ffplay -autoexit out.wav)
```

---

## 7. Best Practices & Pitfalls

**Keep a trainable source** (PyTorch/safetensors or TF) → **export to ONNX** for portability → **compile** to engines (TensorRT/TFLite/Core ML) for targets.  
**For LLMs**, ship **GGUF** via **llama.cpp**; don’t expect to “convert back.”

-   Test with reference inputs **before** converting/quantizing.
-   Beware **op coverage**: custom layers may need replacements (e.g., Swish ≈ SiLU).
-   Quantize with calibration where possible (INT8). For LLMs, try Q4_K/Q5_K variants.
-   Always keep tokenizer files (`tokenizer.json`, `vocab.json`, `tokens.txt`).
-   Benchmark **end‑to‑end latency** (I/O + preprocessing + runtime), not just pure inference.
-   For NVIDIA servers, cache **TensorRT engines** per GPU arch. For desktops, use **ORT**.
-   On mobile, aim for **TFLite/Core ML** early—port last‑mile ops sooner than later.

---

## 8. Glossary

-   **ONNX**: Open Neural Network Exchange — portable graph format for inference.
-   **ONNX Runtime (ORT)**: execution engine for ONNX (CPU/GPU and more).
-   **GGUF**: quantized LLM weight file format for `llama.cpp`.
-   **Engine**: compiled, device‑specific binary for fast inference (e.g., TensorRT `.plan`).
-   **Quantization**: lower precision (INT8/INT4) to reduce memory & latency.
-   **Endpointing**: detecting end of speech (streaming ASR).
-   **EP (Execution Provider)**: hardware backend plugin for ORT (CUDA, DirectML…).

---

## 9. Quick Profiles (Copy‑ready Stacks)

### 9.1 Local Voice Assistant (Jarvis‑feel, zero cloud)

-   **ASR**: sherpa‑onnx streaming (int8) → **partials/finals**
-   **LLM**: Ollama (`llama3.2:3b-instruct` Q4/Q5)
-   **TTS**: Piper (local ONNX voice)
-   **Optional**: Whisper large‑v3 refine (async), sqlite‑vec memory, MCP tools

### 9.2 NVIDIA Server Inference (max perf)

-   Source: PyTorch/safetensors → **ONNX** → **TensorRT engine**
-   Serve: Triton Inference Server (optional)
-   Track engines per GPU arch/driver.

### 9.3 Mobile App

-   Source: PyTorch/TF → **ONNX** → **TFLite** (Android) / **Core ML** (iOS)
-   Use TFLite delegates / Core ML accel; aim INT8 where possible.

---

## 10. Licensing & Safety Notes

-   Check **model licenses** (e.g., LLaMA‑family restrictions; commercial use).
-   Verify **dataset licenses** and usage policies.
-   Respect **export controls** for certain advanced models.
-   For voice apps, disclose recording behavior; store data responsibly.

---

### One‑page Summary (TL;DR)

-   **LLMs** → **GGUF + llama.cpp** (Ollama/LM Studio). Keep a PyTorch source for fine‑tune.
-   **Everything else** (CV/ASR/TTS) → **ONNX + ONNX Runtime**. Convert to **TensorRT/TFLite/Core ML/OpenVINO** as needed.
-   Treat engines & GGUF as **terminal** (one‑way) formats. Keep the **source** & **ONNX** around.
-   Measure **latency end‑to‑end** and plan **quantization** early.
-   Prefer **streaming** ASR for assistants (built‑in endpointing + interims).

---

_Happy building. If you need a CLI that outputs “recommended format/runtime” given task & device, add an issue—we’ll script it._
