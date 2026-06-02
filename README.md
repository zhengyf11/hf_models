# hf_models
从 huggingface 上下载模型权重

## 安装依赖

建议先创建虚拟环境，再安装项目依赖，避免把依赖安装到系统 Python 中。

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果不想使用虚拟环境，也可以在已经由你自己管理的 Python 环境中执行：

```powershell
python -m pip install -r requirements.txt
```

### 常见安装问题

如果安装依赖时遇到 `externally-managed-environment` 或 PEP 668 相关提示，说明当前系统 Python 由操作系统或包管理器托管，不建议直接执行全局 `pip install`。推荐按上面的步骤创建并激活虚拟环境后，再运行：

```bash
python -m pip install -r requirements.txt
```

不要为了绕过该提示直接使用 `--break-system-packages`，除非你明确了解这会修改系统 Python 环境并可能影响系统工具。

## 使用示例

下载整个仓库并在完成后校验文件完整性：

```powershell
python download_hf_weights.py --model=deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro --all-files --verify
```

默认优先从国内镜像站 `https://hf-mirror.com` 下载。需要强制优先使用 Hugging Face 官方站时，指定 `--endpoint`：

```powershell
python download_hf_weights.py --model=deepseek-ai/DeepSeek-V4-Pro -o D:\models\DeepSeek-V4-Pro --all-files --verify --endpoint https://huggingface.co
```

常用参数：

- `--all-files`：下载整个仓库；默认只下载常见模型权重文件，例如 `*.safetensors`、`*.bin`、`*.gguf`、`*.pt`、`*.onnx` 以及分片索引文件。
- `--revision main`：指定分支、tag 或 commit。
- `--endpoint https://huggingface.co`：覆盖首选下载站点；默认优先使用 `https://hf-mirror.com`。
- `--include "*.json"`：额外下载匹配的文件，可重复使用。
- `--exclude "optimizer*"`：排除匹配的文件，可重复使用。
- `--token hf_xxx`：访问私有或受限模型，也可以设置 `HF_TOKEN` 环境变量。
- `--dry-run`：只列出将下载的文件，不实际下载。
- `--verify`：下载完成后检查文件是否存在、大小是否一致；LFS 文件会额外校验 SHA256。
