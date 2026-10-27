# Stage 00 — Three upstream implementations: source-level study

**Pinned source versions:** [sources.lock.json](sources.lock.json). These observations are based on actual named source files (not just README summaries) and are starting hypotheses for an independent design.

## 1. nano-vLLM: minimal system seam

| File | Observed functions | Learning |
| --- | --- | --- |
| `nanovllm/engine/llm_engine.py` | `LLMEngine.add_request`, `step`, `generate` | Offline API binds tokenizer, scheduler and runner. |
| `nanovllm/engine/scheduler.py` (93 lines in pinned snapshot) | `schedule`, `preempt`, `postprocess` | Explicit waiting/running queues, token and sequence budgets, cache admission. |
| `nanovllm/engine/block_manager.py` (121 lines) | `compute_hash`, `allocate`, `may_append`, `deallocate` | Cache block identities, refcounts and prefix hashes make re-use visible. |
| `nanovllm/engine/model_runner.py` | `prepare_prefill`, `prepare_decode`, `allocate_kv_cache`, `run`, `capture_cudagraph` | Logical runner captures batch layouts; CUDA Graph and worker process are runtime details. |

**Transfer:** understandable component boundaries and invariant-oriented tests. **Do not transfer blindly:** global sequence IDs, hashing/isolation policy or implicit mutable runner state, especially for future async scheduling.

## 2. vLLM V1: production separation

| File | Observed functions | Learning |
| --- | --- | --- |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule`, `update_from_output`, `add_request`, `finish_requests` | Scheduler admits both prefills and decode using computed-token state and capacity. |
| `vllm/v1/engine/core.py` | `EngineCore.add_request`, `step`, `step_with_batch_queue`, `shutdown` | Decouples client and worker lifetime; exposes async batch state. |
| `vllm/v1/core/kv_cache_manager.py` | Cache policy / reservation | Future path for hashed prefix blocks and model-type specialization. |

**Transfer:** unified token accounting, separation of scheduler input and worker output, bounded admission and release. **Avoid:** transplanting thousands of lines of feature interaction before minimal correctness.

## 3. DeepInfer: model-specific extreme throughput

| File | Source observation | Learning |
| --- | --- | --- |
| `csrc/engine/scheduler.hpp` (1,330 lines) | C++ scheduler, PyTorch/C10 and pybind-dependent scheduling path | Run dispatch decisions with lower host-side overhead, but introduces C++ deployment complexity. |
| `deepinfer/engine/llm_engine.py` (~1,030 lines) | LLMEngine methods `step`, `_execute_model_async_step`, `flush_inflight_reqs`, `post_process` | Overlap execution and postprocess, include deferred release / inflight tracking. |
| `doc/inference_deployment_qa.md` | Describes EP/EPLB, DSpark, async scheduling, SWA bounded replay and cache retention | Significant model/hardware/system assumptions; not drop-in generic algorithms. |

**Transfer:** conceptual async scheduling/commit, experimental EP/EPLB and speculative budgets; **do not assume** parity with V4.1-Flash kernels or its deployment environment. DeepInfer explicitly says part of its work derived from vLLM; preserve attribution for any copied code.

## Proposed dependency graph

```text
Stage 00: docs + lock + tests
  -> Stage 01: CPU/HF reference -> DenseRunner
  -> Stage 02: BlockManager -> CacheState
  -> Stage 03: Scheduler -> StepPlan -> Commit
  -> Stage 04: ChunkedPrefill + PrefixIndex
  -> Stage 05: optional kernel backend
  -> Stage 06: async executor with versions / acknowledgements
  -> Stage 07–09: experiments (MoE, speculation, SWA)
  -> Stage 10–12: serving, distributed experiment, evaluation
```

## Open questions / explicit investigation backlog

1. Benchmark reference HF vs eager exactness under TF32/AMP and sampling seeds.
2. Cache fingerprint fields (checkpoint revision, tokenizer, rope, quantization, LoRA/tenant).
3. GPU CUDA Graph capture restrictions and graph bucket tuning.
4. Dynamic speculation correctness when rejection and draft stochastic sampling coexist.
5. Multi-rank dispatch ordering correctness and balanced-expert placement cost.
6. How SWA replay interacts with non-SWA state (MLA/other model-specific state).

