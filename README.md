# hf_models
从 huggingface 上下载模型权重

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 使用示例

默认优先从国内镜像站 `https://hf-mirror.com` 下载：

```powershell
python download_hf_weights.py --model=deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro
```

也可以直接传 Hugging Face 官方 URL，脚本仍会默认优先使用镜像站下载：

```powershell
python download_hf_weights.py --model=https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro
```

需要强制优先使用 Hugging Face 官方站时，指定 `--endpoint`：

```powershell
python download_hf_weights.py --model=deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro --endpoint https://huggingface.co
```

默认只下载常见模型权重文件，例如 `*.safetensors`、`*.bin`、`*.gguf`、`*.pt`、`*.onnx` 以及分片索引文件。需要下载整个仓库时使用：

```powershell
python download_hf_weights.py --model=https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro --all-files
```

下载完成后校验文件完整性：

```powershell
python download_hf_weights.py --model=deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro --verify
```

常用参数：

- `--revision main`：指定分支、tag 或 commit。
- `--endpoint https://huggingface.co`：覆盖首选下载站点；默认优先使用 `https://hf-mirror.com`。
- `--include "*.json"`：额外下载匹配的文件，可重复使用。
- `--exclude "optimizer*"`：排除匹配的文件，可重复使用。
- `--token hf_xxx`：访问私有或受限模型，也可以设置 `HF_TOKEN` 环境变量。
- `--dry-run`：只列出将下载的文件，不实际下载。
- `--verify`：下载完成后检查文件是否存在、大小是否一致；LFS 文件会额外校验 SHA256。
