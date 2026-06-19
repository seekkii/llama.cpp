import argparse
import importlib
import sys
from pathlib import Path


def _prefer_build_bin() -> None:
    build_bin = Path(__file__).resolve().parents[2] / "build" / "bin"
    if build_bin.is_dir():
        sys.path.insert(0, str(build_bin))


_prefer_build_bin()


llama_cpp_cli = importlib.import_module("llama_cpp_cli")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-p", "--prompt", default="Hello")
    parser.add_argument("--reasoning", choices=("auto", "on", "off"), default="off")
    args = parser.parse_args()

    llama = llama_cpp_cli.Llama([
        "-m", args.model,
        "--simple-io",
        "--single-turn",
        "--reasoning", args.reasoning,
    ])

    output = llama(args.prompt, max_tokens=128, stream=True)

    for chunk in output:
        choice = chunk["choices"][0]
        text = choice.get("text", "")
        reasoning_text = choice.get("reasoning_text", "")
        if reasoning_text:
            print(reasoning_text, end="", flush=True)
        if text:
            print(text, end="", flush=True)

    print()
    llama.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())