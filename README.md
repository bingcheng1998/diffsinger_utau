# DiffSinger UTAU

DiffSinger UTAU 推理工具包，提供命令行工具和 Python API 进行语音合成。

## 功能特性

- 完整的 DiffSinger 推理流程：时长预测 → 音高预测 → 方差预测 → 声学模型 → 声码器
- 支持多语言和多说话人
- 提供命令行工具和 Python API
- 支持音高移调、性别控制等参数调节

## 安装

### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/diffsinger-utau.git
cd diffsinger-utau

# 创建虚拟环境（推荐使用 Python 3.8）
conda create -n diffsinger python=3.8
conda activate diffsinger

# 安装依赖
pip install -e .
```

### 从 PyPI 安装（待发布）

```bash
pip install diffsinger-utau
```

## 使用方法

### 命令行工具

```bash
# 基本用法
dsutau samples/07_春江花月夜.1.ds

# 指定语音库和参数
dsutau samples/07_春江花月夜.1.ds \
  --voice-bank artifacts/JiangKe_DiffSinger_CE_25.06 \
  --lang zh \
  --speaker "jiangke" \
  --key-shift 4 \
  --pitch-steps 10 \
  --variance-steps 10 \
  --acoustic-steps 50 \
  --gender 0.0 \
  --output output/pred_all
```

### Python API

```python
from pathlib import Path
from voice_bank import PredAll
from voice_bank.commons.ds_reader import DSReader

# 初始化预测器
voice_bank = Path("artifacts/JiangKe_DiffSinger_CE_25.06")
predictor = PredAll(voice_bank)

# 读取 DS 文件
ds = DSReader("samples/07_春江花月夜.1.ds").read_ds()[0]

# 执行完整推理
results = predictor.predict_full_pipeline(
    ds=ds,
    lang="zh",
    speaker=None,  # 随机选择说话人
    key_shift=0,
    pitch_steps=10,
    variance_steps=10,
    acoustic_steps=50,
    gender=0.0,
    output_dir="output/pred_all",
    save_intermediate=True,
)

print(f"生成音频: {results['audio_path']}")
```

## 参数说明

- `--voice-bank`: 语音库目录路径
- `--lang`: 语言代码（默认: zh）
- `--speaker`: 说话人名称（不指定则随机选择）
- `--key-shift`: 音高移调半音数（默认: 0）
- `--pitch-steps`: 音高扩散采样步数（默认: 10）
- `--variance-steps`: 方差扩散采样步数（默认: 10）
- `--acoustic-steps`: 声学模型扩散采样步数（默认: 50）
- `--gender`: 性别参数 [-1, 1]，-1为男性，1为女性（默认: 0）
- `--output`: 输出目录（默认: output/pred_all）
- `--no-intermediate`: 不保存中间结果文件

## 输出文件

推理完成后会在输出目录生成以下文件：

- `step1_duration.ds`: 时长预测结果
- `step2_pitch.ds`: 音高预测结果
- `step3_variance.ds`: 方差预测结果
- `step4_mel_spectrogram.png`: Mel频谱图
- `step4_mel_data.json`: Mel数据（JSON格式）
- `step5_final_audio.wav`: 最终音频文件
- `complete_prediction.ds`: 完整预测结果

## 系统要求

- Python 3.8
- 支持的操作系统：Windows, macOS, Linux
- 内存：建议 8GB 以上
- 存储：至少 2GB 可用空间

## 依赖项

- numpy>=1.21,<1.25
- librosa>=0.9,<0.10
- PyYAML>=6.0
- onnxruntime>=1.12,<1.17
- torch>=1.10,<2.0
- pypinyin>=0.40
- scipy>=1.7

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码格式化
black voice_bank/

# 代码检查
flake8 voice_bank/
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v0.1.0
- 初始版本
- 支持完整的 DiffSinger 推理流程
- 提供命令行工具和 Python API
