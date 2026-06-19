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

The script checks every `*.patch` before applying it.

To apply only selected patches:

```bash
scripts/apply_sglang_patches.sh /mnt/workspace/third_party/sglang-v0.5.12.post1 dllm_trace.patch
```

Current patches:

- `deterministic_dllm_compat.patch`: compatibility patch for SGLang deterministic inference with LLaDA 2.1 dLLM.
- `dllm_trace.patch`: writes JointThreshold M2T/T2T trace events when `trace_path` is set in the DLLM YAML config.
