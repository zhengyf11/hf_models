#!/usr/bin/env python3
"""Download model weight files from Hugging Face or hf-mirror."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


DEFAULT_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.tflite",
    "*safetensors.index.json",
    "*pytorch_model.bin.index.json",
    "*tf_model.h5.index.json",
    "*flax_model.msgpack.index.json",
]

RESERVED_REPO_PATHS = {
    "blob",
    "commits",
    "discussions",
    "resolve",
    "settings",
    "tree",
}


@dataclass(frozen=True)
class ModelLocation:
    repo_id: str
    endpoint: str | None
    revision: str | None = None


def parse_model_location(value: str) -> ModelLocation:
    """Parse a Hugging Face model URL, mirror URL, or plain repo id."""

    text = value.strip()
    if not text:
        raise ValueError("模型 URL 或 repo id 不能为空")

    if "://" not in text:
        host_candidate = text.split("/", 1)[0].lower()
        if host_candidate.startswith("www."):
            host_candidate = host_candidate[4:]
        if host_candidate in {"huggingface.co", "hf-mirror.com"}:
            text = f"https://{text}"

    parsed = urlparse(text)
    if not parsed.scheme and not parsed.netloc:
        return ModelLocation(repo_id=text.strip("/"), endpoint=None)

    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    if host not in {"huggingface.co", "hf-mirror.com"}:
        raise ValueError(
            "仅支持 huggingface.co、hf-mirror.com URL，或直接传入 repo id"
        )

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not parts:
        raise ValueError("URL 中没有找到模型仓库路径")

    if parts[0] in {"datasets", "spaces"}:
        raise ValueError("当前脚本只支持模型仓库 URL，不支持 datasets 或 spaces")

    if len(parts) >= 2 and parts[1] not in RESERVED_REPO_PATHS:
        repo_parts = parts[:2]
        rest = parts[2:]
    else:
        repo_parts = parts[:1]
        rest = parts[1:]

    repo_id = "/".join(repo_parts)
    revision = None
    if len(rest) >= 2 and rest[0] in {"blob", "resolve", "tree"}:
        revision = rest[1]

    endpoint = "https://hf-mirror.com" if host == "hf-mirror.com" else "https://huggingface.co"
    return ModelLocation(repo_id=repo_id, endpoint=endpoint, revision=revision)


def split_patterns(values: Iterable[str]) -> list[str]:
    patterns: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                patterns.append(item)
    return patterns


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 或 hf-mirror 下载模型权重文件。"
    )
    parser.add_argument(
        "model",
        help=(
            "模型 URL 或 repo id，例如 "
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="下载目标目录。文件会直接放到这个目录中。",
    )
    parser.add_argument(
        "-r",
        "--revision",
        help="指定分支、tag 或 commit。默认使用 URL 中的 revision 或仓库默认分支。",
    )
    parser.add_argument(
        "--endpoint",
        help=(
            "覆盖下载站点，例如 https://hf-mirror.com。"
            "传入普通 repo id 时可用这个参数选择镜像源。"
        ),
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="下载仓库中的全部文件，而不是只下载常见权重文件。",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATTERN",
        help='额外包含的文件通配符，可重复使用或用逗号分隔，例如 --include "*.json"',
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help='排除的文件通配符，可重复使用或用逗号分隔，例如 --exclude "optimizer*"',
    )
    parser.add_argument(
        "--token",
        help="访问私有或受限模型时使用的 Hugging Face token；也可以设置 HF_TOKEN 环境变量。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并发下载线程数，默认 8。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载文件。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出将下载的文件，不实际下载。",
    )
    return parser


def print_dry_run_result(result: object) -> None:
    if not isinstance(result, list):
        print(result)
        return

    print(f"将下载 {len(result)} 个文件：")
    for item in result:
        filename = (
            getattr(item, "filename", None)
            or getattr(item, "path", None)
            or getattr(item, "file_path", None)
            or str(item)
        )
        size = getattr(item, "size_on_disk", None) or getattr(item, "size", None)
        if isinstance(size, int):
            print(f"- {filename} ({size / 1024 / 1024:.2f} MiB)")
        else:
            print(f"- {filename}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_workers < 1:
        parser.error("--max-workers 必须大于等于 1")

    try:
        location = parse_model_location(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "缺少依赖 huggingface_hub，请先运行：pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    extra_includes = split_patterns(args.include)
    excludes = split_patterns(args.exclude)
    allow_patterns = None if args.all_files else DEFAULT_WEIGHT_PATTERNS + extra_includes
    ignore_patterns = excludes or None
    endpoint = args.endpoint or location.endpoint
    revision = args.revision or location.revision

    print(f"模型仓库: {location.repo_id}")
    print(f"下载站点: {endpoint or 'huggingface_hub 默认端点'}")
    print(f"目标目录: {output_dir}")
    if revision:
        print(f"Revision: {revision}")
    print("下载范围: 全部文件" if args.all_files else "下载范围: 常见模型权重文件")

    try:
        result = snapshot_download(
            repo_id=location.repo_id,
            repo_type="model",
            revision=revision,
            local_dir=None if args.dry_run else output_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=args.token,
            endpoint=endpoint,
            max_workers=args.max_workers,
            force_download=args.force,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n下载已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"下载失败: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print_dry_run_result(result)
    else:
        print(f"下载完成: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
