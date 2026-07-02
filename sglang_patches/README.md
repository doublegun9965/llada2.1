# SGLang Patches

This directory stores patches for SGLang source changes used by experiments.

Do not commit the SGLang source tree itself. Keep source checkouts under ignored
paths such as:

- local workspace: `third_party/sglang-v0.5.12.post1`
- server workspace: `/mnt/workspace/third_party/sglang-v0.5.12.post1`

## Create a Patch

After editing files inside the SGLang checkout:

```bash
scripts/save_sglang_patch.sh third_party/sglang-v0.5.12.post1 my_experiment.patch
```

Then commit the generated file under `sglang_patches/`.

## Apply Patches on the Server

From this repository:

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1
```

The script checks and applies each `*.patch` sequentially.

To apply only selected patches:

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1 dllm_trace.patch
```

Current patches:

- `deterministic_dllm_compat.patch`: compatibility patch for SGLang deterministic inference with LLaDA 2.1 dLLM.
- `dllm_trace.patch`: writes JointThreshold M2T/T2T trace events when `trace_path` is set in the DLLM YAML config.
- `joint_threshold_t2t_fallback.patch`: after `dllm_trace.patch`, adds optional post-mask T2T top-k fallback ranked by positive replacement advantage.
- `t2t_advantage_threshold.patch`: after the fallback patch, lets normal T2T thresholding use either proposed-token probability or replacement logit advantage.
- `rocm_disable_vllm_rmsnorm.patch`: makes `SGLANG_DISABLE_VLLM_RMSNORM=1` disable the incompatible ROCm vLLM RMSNorm path.

The default alphabetical application order is intentional: `dllm_trace.patch` must be applied before `joint_threshold_t2t_fallback.patch`, and `t2t_advantage_threshold.patch` must be applied after both.
