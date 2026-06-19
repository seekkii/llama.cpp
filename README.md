# `llama.cpp` for AMD Instinct MI50 (`gfx906`)

A performance-focused fork of `llama.cpp` for AMD Instinct MI50, support newer model architectures with backported `gfx906` kernel optimizations.

## Highlights

* Support for newer model architectures, including Gemma 4, and Qwen 3.5
* Backported `gfx906` optimizations from `llama.cpp-gfx906`
* Optimized MMVQ/MMQ paths for `Q4_0`, `Q4_1`, and `Q8_0`
* MI50-tuned FlashAttention and LDS improvements

## Build

### Linux

```bash
chmod +x SCRIPT_compile_MI50
./SCRIPT_compile_MI50
```

## Notes
* Exerimental build options and configuration variables are detailed via comments inside `SCRIPT_compile_MI50`.*
* Newer `K-quants` may fall back to slower generic HIP paths

## Acknowledgments

* Base: `ggml-org/llama.cpp`
* Kernel optimizations: `iacopPBK/llama.cpp-gfx906`