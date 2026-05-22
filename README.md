# hf_models
从 huggingface 上下载模型权重

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 使用示例

从 Hugging Face 官方站下载：

```powershell
python download_hf_weights.py https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro
```

从国内镜像站下载：

```powershell
python download_hf_weights.py https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro
```

也可以直接传 repo id，并指定镜像端点：

```powershell
python download_hf_weights.py deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro --endpoint https://hf-mirror.com
```

默认只下载常见模型权重文件，例如 `*.safetensors`、`*.bin`、`*.gguf`、`*.pt`、`*.onnx` 以及分片索引文件。需要下载整个仓库时使用：

```powershell
python download_hf_weights.py https://hf-mirror.com/deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro --all-files
```

常用参数：

- `--revision main`：指定分支、tag 或 commit。
- `--include "*.json"`：额外下载匹配的文件，可重复使用。
- `--exclude "optimizer*"`：排除匹配的文件，可重复使用。
- `--token hf_xxx`：访问私有或受限模型，也可以设置 `HF_TOKEN` 环境变量。
- `--dry-run`：只列出将下载的文件，不实际下载。
