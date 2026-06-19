from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLI_BINARY = REPO_ROOT / "build" / "bin" / "llama-cli"
DEFAULT_MODULE_DIR = REPO_ROOT / "build" / "bin"
DEFAULT_MODES = [
    "cli-subprocess",
    "binding-run",
    "binding-engine-fresh",
    "binding-engine-reuse",
]


@contextmanager
def suppress_process_output() -> Any:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


def prefer_module_dir(module_dir: Path) -> None:
    sys.path.insert(0, str(module_dir))


def load_binding_module(module_dir: Path):
    prefer_module_dir(module_dir)
    try:
        return importlib.import_module("llama_cpp_cli")
    except ImportError:
        from llama_cpp_cli_loader import llama_cpp_cli
        return llama_cpp_cli


def build_base_args(model: str, extra_args: list[str]) -> list[str]:
    return ["-m", model, *extra_args]


def build_prompt_args(prompt: str, max_tokens: int) -> list[str]:
    return ["--single-turn", "-p", prompt, "-n", str(max_tokens)]


def measure_cli_subprocess(cli_binary: Path, base_args: list[str], prompt: str, max_tokens: int) -> dict[str, Any]:
    cmd = [str(cli_binary), *base_args, *build_prompt_args(prompt, max_tokens)]
    start = time.perf_counter()
    completed = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    total = time.perf_counter() - start
    return {
        "mode": "cli-subprocess",
        "returncode": completed.returncode,
        "total_sec": total,
    }


def measure_binding_run(module_dir: Path, base_args: list[str], prompt: str, max_tokens: int) -> dict[str, Any]:
    llama_cpp_cli = load_binding_module(module_dir)
    start = time.perf_counter()
    with suppress_process_output():
        rc = llama_cpp_cli.run([*base_args, *build_prompt_args(prompt, max_tokens)])
    total = time.perf_counter() - start
    return {
        "mode": "binding-run",
        "returncode": rc,
        "total_sec": total,
    }


def measure_binding_engine_fresh(module_dir: Path, base_args: list[str], prompt: str, max_tokens: int) -> dict[str, Any]:
    llama_cpp_cli = load_binding_module(module_dir)

    with suppress_process_output():
        start_total = time.perf_counter()
        start_load = time.perf_counter()
        engine = llama_cpp_cli.Llama(base_args)
        load_sec = time.perf_counter() - start_load

        start_request = time.perf_counter()
        result = engine(prompt, max_tokens=max_tokens, stream=False)
        request_sec = time.perf_counter() - start_request
        engine.close()
        total_sec = time.perf_counter() - start_total

    return {
        "mode": "binding-engine-fresh",
        "returncode": 0,
        "load_sec": load_sec,
        "request_sec": request_sec,
        "total_sec": total_sec,
        "text_len": len(result["choices"][0]["text"]),
    }


def measure_binding_engine_reuse(
    module_dir: Path,
    base_args: list[str],
    prompt: str,
    max_tokens: int,
    reuse_requests: int,
) -> dict[str, Any]:
    llama_cpp_cli = load_binding_module(module_dir)

    request_times: list[float] = []
    text_lens: list[int] = []
    with suppress_process_output():
        start_load = time.perf_counter()
        engine = llama_cpp_cli.Llama(base_args)
        load_sec = time.perf_counter() - start_load

        for _ in range(reuse_requests):
            engine.clear()
            start_request = time.perf_counter()
            result = engine(prompt, max_tokens=max_tokens, stream=False)
            request_times.append(time.perf_counter() - start_request)
            text_lens.append(len(result["choices"][0]["text"]))

        engine.close()

    return {
        "mode": "binding-engine-reuse",
        "returncode": 0,
        "load_sec": load_sec,
        "request_sec": statistics.mean(request_times),
        "request_sec_min": min(request_times),
        "request_sec_max": max(request_times),
        "reuse_requests": reuse_requests,
        "text_len": statistics.mean(text_lens) if text_lens else 0,
        "total_sec": load_sec + sum(request_times),
    }


def worker(args: argparse.Namespace) -> int:
    extra_args = shlex.split(args.extra_args)
    base_args = build_base_args(args.model, extra_args)
    module_dir = Path(args.module_dir)
    cli_binary = Path(args.cli_binary)

    if args.worker_mode == "cli-subprocess":
        result = measure_cli_subprocess(cli_binary, base_args, args.prompt, args.max_tokens)
    elif args.worker_mode == "binding-run":
        result = measure_binding_run(module_dir, base_args, args.prompt, args.max_tokens)
    elif args.worker_mode == "binding-engine-fresh":
        result = measure_binding_engine_fresh(module_dir, base_args, args.prompt, args.max_tokens)
    elif args.worker_mode == "binding-engine-reuse":
        result = measure_binding_engine_reuse(module_dir, base_args, args.prompt, args.max_tokens, args.reuse_requests)
    else:
        raise ValueError(f"unknown worker mode: {args.worker_mode}")

    print(json.dumps(result))
    return 0 if result.get("returncode", 1) == 0 else result["returncode"]


def parse_worker_output(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"worker produced no output:\n{completed.stderr}")
    for line in lines:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(
        "worker did not emit a JSON result line\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def spawn_worker(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--module-dir",
        args.module_dir,
        "--cli-binary",
        args.cli_binary,
        "--extra-args",
        args.extra_args,
        "--reuse-requests",
        str(args.reuse_requests),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker mode '{mode}' failed with code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_worker_output(completed)


def summarize(mode: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mode": mode,
        "runs": len(results),
        "avg_total_sec": statistics.mean(r["total_sec"] for r in results),
        "min_total_sec": min(r["total_sec"] for r in results),
        "max_total_sec": max(r["total_sec"] for r in results),
    }

    if all("load_sec" in r for r in results):
        summary["avg_load_sec"] = statistics.mean(r["load_sec"] for r in results)
    if all("request_sec" in r for r in results):
        summary["avg_request_sec"] = statistics.mean(r["request_sec"] for r in results)
    if all("request_sec_min" in r for r in results):
        summary["avg_request_sec_min"] = statistics.mean(r["request_sec_min"] for r in results)
    if all("request_sec_max" in r for r in results):
        summary["avg_request_sec_max"] = statistics.mean(r["request_sec_max"] for r in results)

    return summary


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print("\nBenchmark summary")
    print("=" * 100)
    header = (
        f"{'mode':<24} {'runs':>4} {'avg_total_s':>12} {'min_total_s':>12} "
        f"{'max_total_s':>12} {'avg_load_s':>12} {'avg_req_s':>12}"
    )
    print(header)
    print("-" * len(header))
    for item in summaries:
        print(
            f"{item['mode']:<24} {item['runs']:>4d} "
            f"{item['avg_total_sec']:>12.3f} {item['min_total_sec']:>12.3f} {item['max_total_sec']:>12.3f} "
            f"{item.get('avg_load_sec', float('nan')):>12.3f} {item.get('avg_request_sec', float('nan')):>12.3f}"
        )
    print("=" * 100)
    print("Notes:")
    print("- `cli-subprocess` measures the native `llama-cli` executable end-to-end.")
    print("- `binding-run` measures `llama_cpp_cli.run(...)`, which follows the CLI main-style path.")
    print("- `binding-engine-fresh` measures `Llama(...)` construction plus one completion in a fresh engine.")
    print("- `binding-engine-reuse` measures one model load plus repeated completions on the same engine.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark overhead between native llama-cli and the nanobind Python bindings.",
    )
    parser.add_argument("--model", required=True, help="Path to the GGUF model.")
    parser.add_argument("--prompt", default="hi", help="Prompt to run for each benchmark.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to request per completion.")
    parser.add_argument("--runs", type=int, default=3, help="Number of isolated runs per benchmark mode.")
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated benchmark modes: cli-subprocess,binding-run,binding-engine-fresh,binding-engine-reuse",
    )
    parser.add_argument(
        "--extra-args",
        default="--simple-io",
        help="Extra llama-cli arguments passed after the model path, parsed with shell-style splitting.",
    )
    parser.add_argument("--reuse-requests", type=int, default=3, help="Number of requests to issue in `binding-engine-reuse` mode.")
    parser.add_argument("--cli-binary", default=str(DEFAULT_CLI_BINARY), help="Path to the native `llama-cli` binary.")
    parser.add_argument("--module-dir", default=str(DEFAULT_MODULE_DIR), help="Directory containing `llama_cpp_cli*.so`.")
    parser.add_argument("--worker-mode", choices=DEFAULT_MODES, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker_mode:
        return worker(args)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}

    for mode in modes:
        print(f"Running {mode} ...")
        for _ in range(args.runs):
            results_by_mode[mode].append(spawn_worker(args, mode))

    summaries = [summarize(mode, results) for mode, results in results_by_mode.items()]
    print_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
