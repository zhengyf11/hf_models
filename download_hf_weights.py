#!/usr/bin/env python3
"""Download model weight files from Hugging Face or hf-mirror."""

import argparse
import fnmatch
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen


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

DEFAULT_ENDPOINT = "https://hf-mirror.com"
FALLBACK_ENDPOINT = "https://huggingface.co"
HASH_CHUNK_SIZE = 16 * 1024 * 1024

RESERVED_REPO_PATHS = {
    "blob",
    "commits",
    "discussions",
    "resolve",
    "settings",
    "tree",
}


class ModelLocation:
    def __init__(self, repo_id, endpoint, revision=None):
        self.repo_id = repo_id
        self.endpoint = endpoint
        self.revision = revision

    def __repr__(self):
        return (
            "ModelLocation(repo_id={!r}, endpoint={!r}, revision={!r})".format(
                self.repo_id, self.endpoint, self.revision
            )
        )


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

    endpoint = "https://{}".format(host)
    return ModelLocation(repo_id=repo_id, endpoint=endpoint, revision=revision)


def split_patterns(values: Iterable[str]) -> List[str]:
    patterns = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                patterns.append(item)
    return patterns


def endpoint_candidates(primary_endpoint, model_endpoint=None):
    endpoint = (primary_endpoint or DEFAULT_ENDPOINT).rstrip("/")
    endpoints = [endpoint]
    if model_endpoint and model_endpoint.rstrip("/") not in endpoints:
        endpoints.append(model_endpoint.rstrip("/"))

    fallback = FALLBACK_ENDPOINT if endpoint == DEFAULT_ENDPOINT else DEFAULT_ENDPOINT
    if fallback not in endpoints:
        endpoints.append(fallback)
    return endpoints


def http_error_message(exc):
    if isinstance(exc, HTTPError):
        return "HTTP {} {}".format(exc.code, exc.reason)
    if isinstance(exc, URLError):
        return str(exc.reason)
    return str(exc)


def read_json_url(url, token=None, timeout=60):
    headers = {"User-Agent": "hf-model-weight-downloader"}
    if token:
        headers["Authorization"] = "Bearer {}".format(token)
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def list_repo_files_http(endpoint, repo_id, revision=None, token=None):
    revision = revision or "main"
    url = "{}/api/models/{}/tree/{}?recursive=1".format(
        endpoint.rstrip("/"),
        quote(repo_id.strip("/"), safe="/"),
        quote(revision, safe=""),
    )
    data = read_json_url(url, token=token)
    if isinstance(data, dict):
        message = data.get("error") or data.get("message") or data
        raise RuntimeError("无法读取模型文件列表: {}".format(message))

    files = []
    for item in data:
        if item.get("type") == "file":
            file_info = {
                "path": item.get("path", ""),
                "size": item.get("size"),
                "oid": item.get("oid"),
            }
            if item.get("lfs"):
                file_info["lfs"] = item.get("lfs")
            files.append(file_info)
    files.sort(key=lambda item: item["path"])
    return files


def filter_files(files, allow_patterns=None, ignore_patterns=None):
    selected = []
    for item in files:
        path = item["path"]
        if allow_patterns and not any(fnmatch.fnmatch(path, pattern) for pattern in allow_patterns):
            continue
        if ignore_patterns and any(fnmatch.fnmatch(path, pattern) for pattern in ignore_patterns):
            continue
        selected.append(item)
    return selected


def file_url(endpoint, repo_id, revision, path):
    revision = revision or "main"
    return "{}/{}/resolve/{}/{}".format(
        endpoint.rstrip("/"),
        quote(repo_id.strip("/"), safe="/"),
        quote(revision, safe=""),
        quote(path, safe="/"),
    )


def download_one_http(endpoint, repo_id, revision, item, output_dir, token=None):
    path = item["path"]
    size = item.get("size")
    destination = output_dir / Path(path)
    if destination.exists() and isinstance(size, int) and destination.stat().st_size == size:
        return "跳过已存在文件: {}".format(path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_name(destination.name + ".part")
    headers = {"User-Agent": "hf-model-weight-downloader"}
    if token:
        headers["Authorization"] = "Bearer {}".format(token)
    request = Request(file_url(endpoint, repo_id, revision, path), headers=headers)

    with urlopen(request, timeout=120) as response:
        with temp_destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    os.replace(str(temp_destination), str(destination))
    return "下载完成: {}".format(path)


def download_files_http(endpoint, repo_id, revision, files, output_dir, token=None, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                download_one_http, endpoint, repo_id, revision, item, output_dir, token
            )
            for item in files
        ]
        for future in as_completed(futures):
            print(future.result())


def expected_file_size(item):
    size = item.get("size")
    if isinstance(size, int):
        return size

    lfs = item.get("lfs") or {}
    size = lfs.get("size")
    if isinstance(size, int):
        return size
    return None


def expected_sha256(item):
    lfs = item.get("lfs") or {}
    oid = lfs.get("oid")
    if not isinstance(oid, str):
        return None
    if oid.startswith("sha256:"):
        oid = oid.split(":", 1)[1]
    oid = oid.lower()
    if len(oid) != 64:
        return None
    if any(char not in "0123456789abcdef" for char in oid):
        return None
    return oid


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_downloaded_files(files, output_dir, max_hash_checks=None):
    missing = []
    size_mismatches = []
    hash_mismatches = []
    checked_hashes = 0
    skipped_hashes = 0

    print("开始校验下载文件...")
    for item in files:
        relative_path = item["path"]
        destination = output_dir / Path(relative_path)
        if not destination.exists():
            missing.append(relative_path)
            continue
        if not destination.is_file():
            missing.append(relative_path)
            continue

        expected_size = expected_file_size(item)
        if isinstance(expected_size, int):
            actual_size = destination.stat().st_size
            if actual_size != expected_size:
                size_mismatches.append((relative_path, expected_size, actual_size))
                continue

        expected_hash = expected_sha256(item)
        if expected_hash:
            if max_hash_checks is not None and checked_hashes >= max_hash_checks:
                skipped_hashes += 1
                continue
            actual_hash = sha256_file(destination)
            checked_hashes += 1
            if actual_hash != expected_hash:
                hash_mismatches.append((relative_path, expected_hash, actual_hash))

    if missing:
        print("缺失文件:")
        for path in missing:
            print(f"- {path}")

    if size_mismatches:
        print("大小不一致:")
        for path, expected, actual in size_mismatches:
            print(f"- {path}: expected {expected}, actual {actual}")

    if hash_mismatches:
        print("SHA256 不一致:")
        for path, expected, actual in hash_mismatches:
            print(f"- {path}: expected {expected}, actual {actual}")

    if missing or size_mismatches or hash_mismatches:
        print("校验失败。")
        return False

    print(
        "校验通过: {} 个文件存在且大小一致，{} 个 LFS 文件 SHA256 一致。".format(
            len(files), checked_hashes
        )
    )
    if skipped_hashes:
        print(f"跳过 SHA256 校验: {skipped_hashes} 个文件。")
    return True


def load_selected_files(endpoint, repo_id, revision, token, allow_patterns, ignore_patterns):
    files = list_repo_files_http(endpoint, repo_id, revision=revision, token=token)
    return filter_files(files, allow_patterns, ignore_patterns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 或 hf-mirror 下载模型权重文件。"
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="URL_OR_REPO_ID",
        help=(
            "模型 URL 或 repo id，例如 "
            "https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Pro"
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
            "覆盖首选下载站点，默认 https://hf-mirror.com；"
            "例如 --endpoint https://huggingface.co 可强制优先官方站。"
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="下载完成后校验文件是否存在、大小是否一致；LFS 文件会额外校验 SHA256。",
    )
    return parser


def print_dry_run_result(result) -> None:
    if not isinstance(result, list):
        print(result)
        return

    print(f"将下载 {len(result)} 个文件：")
    for item in result:
        if isinstance(item, dict):
            filename = item.get("path") or item.get("filename") or str(item)
            size = item.get("size")
        else:
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_workers < 1:
        parser.error("--max-workers 必须大于等于 1")

    try:
        location = parse_model_location(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    extra_includes = split_patterns(args.include)
    excludes = split_patterns(args.exclude)
    allow_patterns = None if args.all_files else DEFAULT_WEIGHT_PATTERNS + extra_includes
    ignore_patterns = excludes or None
    endpoint = args.endpoint or DEFAULT_ENDPOINT
    revision = args.revision or location.revision
    token = args.token or os.environ.get("HF_TOKEN")
    endpoints = endpoint_candidates(endpoint, location.endpoint)

    print(f"模型仓库: {location.repo_id}")
    print(f"下载站点: {endpoints[0]}")
    if len(endpoints) > 1:
        print(f"备用站点: {endpoints[1]}")
    print(f"目标目录: {output_dir}")
    if revision:
        print(f"Revision: {revision}")
    print("下载范围: 全部文件" if args.all_files else "下载范围: 常见模型权重文件")
    if args.verify:
        print("完整性校验: 开启")

    if args.dry_run:
        if args.verify:
            print("提示: --dry-run 不下载文件，因此不会执行 --verify 校验。")
        last_error = None
        for candidate in endpoints:
            try:
                result = load_selected_files(
                    candidate,
                    location.repo_id,
                    revision,
                    token,
                    allow_patterns,
                    ignore_patterns,
                )
                if candidate != endpoints[0]:
                    print(f"已切换到备用站点: {candidate}")
                print_dry_run_result(result)
                return 0
            except Exception as exc:
                last_error = exc
                print(
                    "读取文件列表失败，站点 {}: {}".format(
                        candidate, http_error_message(exc)
                    ),
                    file=sys.stderr,
                )
        print(f"下载失败: {last_error}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "缺少依赖 huggingface_hub，将使用标准库 HTTP 下载回退实现。",
            file=sys.stderr,
        )
        last_error = None
        for candidate in endpoints:
            try:
                files = list_repo_files_http(
                    candidate, location.repo_id, revision=revision, token=token
                )
                selected = filter_files(files, allow_patterns, ignore_patterns)
                if candidate != endpoints[0]:
                    print(f"已切换到备用站点: {candidate}")
                download_files_http(
                    candidate,
                    location.repo_id,
                    revision,
                    selected,
                    output_dir,
                    token=token,
                    max_workers=args.max_workers,
                )
                if args.verify and not verify_downloaded_files(selected, output_dir):
                    return 1
                print(f"下载完成: {output_dir}")
                return 0
            except Exception as exc:
                last_error = exc
                print(
                    "HTTP 下载失败，站点 {}: {}".format(
                        candidate, http_error_message(exc)
                    ),
                    file=sys.stderr,
                )
        print(f"下载失败: {last_error}", file=sys.stderr)
        return 1

    try:
        last_error = None
        for candidate in endpoints:
            try:
                result = snapshot_download(
                    repo_id=location.repo_id,
                    repo_type="model",
                    revision=revision,
                    local_dir=output_dir,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    token=token,
                    endpoint=candidate,
                    max_workers=args.max_workers,
                    force_download=args.force,
                )
                if candidate != endpoints[0]:
                    print(f"已切换到备用站点: {candidate}")
                if args.verify:
                    selected = load_selected_files(
                        candidate,
                        location.repo_id,
                        revision,
                        token,
                        allow_patterns,
                        ignore_patterns,
                    )
                    if not verify_downloaded_files(selected, output_dir):
                        return 1
                print(f"下载完成: {result}")
                return 0
            except Exception as exc:
                last_error = exc
                print(
                    "下载失败，站点 {}: {}".format(candidate, http_error_message(exc)),
                    file=sys.stderr,
                )
        print(f"下载失败: {last_error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n下载已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
