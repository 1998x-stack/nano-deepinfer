# Upstream sources and implementation provenance

Pinned at **2026-10-27** via GitHub commit API; exact source content reviewed by repository path.

| Project | Fixed commit | License | Evidence |
| --- | --- | --- | --- |
| nano-vLLM | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` | MIT | [engine files](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine) |
| vLLM V1 | `ace90aeb247a757d32fd8e37e02eb60ba8fd0a46` | Apache-2.0 | [core scheduler](https://github.com/vllm-project/vllm/blob/ace90aeb247a757d32fd8e37e02eb60ba8fd0a46/vllm/v1/core/sched/scheduler.py) |
| DeepInfer | `553741557a57df44a46fb3ab72b6aafeaf945337` | Apache-2.0 | [C++ scheduler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/scheduler.hpp) |

Technical Q&A: [DeepInfer deployment](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md). The released DeepInfer says it provides **offline inference**, primarily for **DeepSeek-V4.1-Flash** on specific hardware. Separate commercial or internal KV-cache storage, P/D production networking and cross-replica serving cannot be assumed to be in this repository.

**Licensing:** Source files studied are not copied. If importing code later, include original copyright, license notices, any NOTICE modifications and source linkage in the importing PR. This repository is Apache-2.0 licensed; upstream nano-vLLM MIT copyright remains owned by its authors.

Detailed analysis: [upstream-analysis](upstream-analysis.md). Programmatic lock: [sources.lock.json](sources.lock.json).

