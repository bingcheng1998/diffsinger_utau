#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys

from .pred_all import PredAll
from .commons.ds_reader import DSReader


def build_parser():
    parser = argparse.ArgumentParser(
        prog="dsutau",
        description="DiffSinger UTAU 推理命令行工具"
    )
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
        predictor.predict_full_pipeline(
            ds=ds,
            lang=args.lang,
            speaker=args.speaker,
            key_shift=args.key_shift,
            pitch_steps=args.pitch_steps,
            variance_steps=args.variance_steps,
            acoustic_steps=args.acoustic_steps,
            gender=args.gender,
            output_dir=args.output,
            save_intermediate=args.save_intermediate,
        )
    except Exception as e:
        print(f"错误: 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


