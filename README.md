# DiffSinger UTAU

DiffSinger UTAU 推理工具包，提供命令行工具和 Python API 进行语音合成。
基于 [diffsinger](https://github.com/openvpi/DiffSinger) 项目，兼容 OpenUtau 声库。

## 功能特性

- 完整的 DiffSinger 推理流程：时长预测 → 音高预测 → 方差预测 → 声学模型 → 声码器
- 支持多语言和多说话人
- 提供命令行工具和 Python API
- 支持音高移调、性别控制等参数调节

## 安装

由于[历史原因](https://github.com/openvpi/DiffSinger/blob/main/docs/GettingStarted.md#deployment)，强依赖 PyTorch 1.13，因此建议使用 Python 3.8。

### 从 PyPI 安装

```bash
pip install diffsinger-utau
```
### 从源码安装

```bash
# 克隆仓库
git clone https://github.com/bingcheng1998/diffsinger_utau.git
cd diffsinger-utau

# 创建虚拟环境（推荐使用 Python 3.8）
conda create -n diffsinger python=3.8
conda activate diffsinger

# 安装依赖
pip install -e .
```

## 使用方法

### 下载声库

什么是声库？声库可以理解为歌唱者的模型，有着各自的音色等特性。

社区提供了[DiffSinger自制声库分享](https://docs.qq.com/sheet/DQXNDY0pPaEpOc3JN)，如果你不确定下载哪个，推荐从[zhibin club](https://www.zhibin.club/)下载[姜柯JiangKe](https://pan.quark.cn/s/254f030af8cb#/list/share/0929019064004907b7b95212c03066ed)声库开始尝试。

下载声库后，需要解压，解压缩后的路径可以作为程序参数进行推理。

### 下载示例 ds 文件

什么是 ds 文件？ds 文件是修改后缀后的标准json文件，内容为歌曲的内容，包含歌词、音高等内容。

社区提供了[示例文件](https://github.com/openvpi/DiffSinger/tree/main/samples)，建议从示例文件推理开始尝试。

### 命令行工具

```bash
# 基本用法
dsutau diffsinger_utau/samples/07_春江花月夜.ds

# 指定语音库和参数
dsutau diffsinger_utau/samples/07_春江花月夜.ds \
  --voice-bank /Users/bc/Music/Singers/Junninghua_v1.4.0_DiffSinger_OpenUtau  \
  --lang zh \
  --speaker "jiangke" \
  --key-shift 2 \
  --pitch-steps 10 \
  --variance-steps 10 \
  --acoustic-steps 20 \
  --gender 0.0 \
  --output output/pred_all
```

### Python API

```python
from pathlib import Path
from diffsinger_utau.voice_bank import PredAll
from diffsinger_utau.voice_bank.commons.ds_reader import DSReader

# 初始化预测器
voice_bank = Path("artifacts/JiangKe_DiffSinger_CE_25.06")
predictor = PredAll(voice_bank)

# 读取 DS 文件
ds = DSReader("samples/07_春江花月夜.ds").read_ds()[0]

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
- `--gender`: 性别参数 [-1, 1]，-1为男性化，1为女性化（默认: 0）
- `--output`: 输出目录（默认: output/pred_all）
- `--no-intermediate`: 不保存中间结果文件

## 输出文件

推理完成后会在输出目录生成以下文件：

- `step1_duration.ds`: 时长预测结果
- `step2_pitch.ds`: 音高预测结果
- `step3_variance.ds`: 方差预测结果
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

### v0.1.2
- 兼容并测试通过声库[JiangKe, LuoXi, YunYe, ZhiBin](https://pan.quark.cn/s/254f030af8cb#/list/share/0929019064004907b7b95212c03066ed)

### v0.1.0
- 初始版本
- 支持完整的 DiffSinger 推理流程
- 提供命令行工具和 Python API
