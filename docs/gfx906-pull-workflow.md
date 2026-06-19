# GFX906 pull workflow

This tree carries local gfx906 integration files that should not be overwritten
by normal upstream pulls.

## One-time setup

Run:

```bash
./scripts/setup-gfx906-merge-driver.sh
```

This configures a local git merge driver named `keep-gfx906`, enables `rerere`,
and sets `pull.rebase=false` so merge-based pulls use the protected file rules
from `.gitattributes`.

## Recommended update command

Use:

```bash
./scripts/gfx906-safe-pull.sh
```

This performs the setup automatically and then runs `git pull --no-rebase`.

## Protected files

The protected set includes:

- `ggml/src/ggml-cuda/gfx906/**`
- `ggml/src/ggml-cuda/common.cuh`
- `ggml/src/ggml-cuda/fattn-common.cuh`
- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/CMakeLists.txt`
- `ggml/src/ggml-hip/CMakeLists.txt`
- `SCRIPT_compile_MI50.sh`
- `SCRIPT_launch_server_MI50.sh`
- `SCRIPT_llama_bench.sh`
- `SCRIPT_llama_bench2.sh`
- `SCRIPT_overclock_upp_MI50.sh`

If you add another local gfx906 integration touchpoint later, add it to
`.gitattributes` with `merge=keep-gfx906`.