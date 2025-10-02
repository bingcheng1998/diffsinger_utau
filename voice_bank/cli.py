#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import numpy as np

from .pred_all import PredAll
from .commons.ds_reader import DSReader
from . import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="dsutau",
        description="DiffSinger UTAU 推理命令行工具"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("ds", type=str, help="输入的 .ds 文件路径")
    parser.add_argument("--voice-bank", dest="voice_bank", type=str,
                        default="artifacts/JiangKe_DiffSinger_CE_25.06",
                        help="语音库目录路径")
    parser.add_argument("--lang", type=str, default="zh", help="语言代码，默认 zh")
    parser.add_argument("--speaker", type=str, default=None, help="说话人名称，不填则随机")
    parser.add_argument("--key-shift", dest="key_shift", type=int, default=0, help="移调半音")
    parser.add_argument("--pitch-steps", dest="pitch_steps", type=int, default=10, help="音高扩散步数")
    parser.add_argument("--variance-steps", dest="variance_steps", type=int, default=10, help="方差扩散步数")
    parser.add_argument("--acoustic-steps", dest="acoustic_steps", type=int, default=50, help="声学扩散步数")
    parser.add_argument("--gender", type=float, default=0.0, help="性别参数 [-1,1]")
    parser.add_argument("--output", type=str, default="output/pred_all", help="输出目录")
    parser.add_argument("--no-intermediate", dest="save_intermediate", action="store_false", help="不保存中间文件")
    return parser


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    voice_bank_path = Path(args.voice_bank)
    if not voice_bank_path.exists():
        print(f"错误: 语音库路径不存在: {voice_bank_path}")
        sys.exit(1)

    ds_path = Path(args.ds)
    if not ds_path.exists():
        print(f"错误: DS文件不存在: {ds_path}")
        sys.exit(1)

    try:
        predictor = PredAll(voice_bank_path)
    except Exception as e:
        print(f"错误: 无法加载语音库: {e}")
        sys.exit(1)

    try:
        ds_reader = DSReader(ds_path)
        sections = ds_reader.read_ds()
        if not sections:
            print(f"错误: 无法读取DS文件: {ds_path}")
            sys.exit(1)
        ds = sections[0]
    except Exception as e:
        print(f"错误: 读取DS文件失败: {e}")
        sys.exit(1)

    try:
        # 逐段预测并根据 offset 混合为单个音频
        sample_rate = predictor.dsvocoder.sample_rate
        section_wavs = []
        section_offsets = []

        output_root = Path(args.output)
        output_root.mkdir(parents=True, exist_ok=True)

        print(f"共读取到 {len(sections)} 个段落，将逐个预测并进行混合...")
        for idx, ds in enumerate(sections):
            try:
                sec_output_dir = output_root / f"section_{idx:03d}"
                results = predictor.predict_full_pipeline(
                    ds=ds,
                    lang=args.lang,
                    speaker=args.speaker,
                    key_shift=args.key_shift,
                    pitch_steps=args.pitch_steps,
                    variance_steps=args.variance_steps,
                    acoustic_steps=args.acoustic_steps,
                    gender=args.gender,
                    output_dir=str(sec_output_dir),
                    save_intermediate=args.save_intermediate,
                )
                wav = results.get('wav')
                if wav is None:
                    raise RuntimeError("未得到音频结果 'wav'")
                # offset 字段为秒
                offset_sec = float(ds.get('offset')) if ds.get('offset') is not None else 0.0
                section_wavs.append(wav.astype(np.float32))
                section_offsets.append(offset_sec)
                print(f"段落 {idx} 完成，offset={offset_sec:.3f}s，长度={len(wav)/sample_rate:.2f}s")
            except Exception as e:
                print(f"错误: 段落 {idx} 推理失败: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        # 混合所有段落
        if not section_wavs:
            print("错误: 没有可混合的音频段")
            sys.exit(1)

        # 计算混合后的总长度（样本数）
        max_len = 0
        start_samples = []
        for wav, offset in zip(section_wavs, section_offsets):
            start = int(round(offset * sample_rate))
            end = start + len(wav)
            start_samples.append(start)
            if end > max_len:
                max_len = end

        mix = np.zeros(max_len, dtype=np.float32)
        for wav, start in zip(section_wavs, start_samples):
            end = start + len(wav)
            mix[start:end] += wav

        # 避免溢出，进行简单限幅
        mix = np.clip(mix, -1.0, 1.0)

        final_path = output_root / "step5_final_audio.wav"
        predictor.pred_vocoder.save_wav(mix, final_path)
        print(f"混合完成，已保存到: {final_path}")

    except Exception as e:
        print(f"错误: 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


