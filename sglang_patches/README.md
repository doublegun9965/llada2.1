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
