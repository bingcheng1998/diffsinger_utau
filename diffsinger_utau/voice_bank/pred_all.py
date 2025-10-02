#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DiffSinger完整预测流程
按照 duration->pitch->variance->acoustic->vocoder 的顺序执行完整的TTS推理流程
"""

import json
import sys
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .commons.voice_bank_reader import VoiceBankReader
from .commons.ds_reader import DSReader
from .commons.utils import save_mel_and_f0_as_json, resample_align_curve

# 导入各个预测器
from .pred_duration import PredDuration
from .pred_pitch import PredPitch
from .pred_variance import PredVariance
from .pred_acoustic import PredAcoustic
from .pred_vocoder import PredVocoder


class PredAll:
    """完整的DiffSinger预测流程"""
    
    def __init__(self, voice_bank_path: Path):
        """
        初始化完整预测流程
        
        Args:
            voice_bank_path: 语音库路径
        """
        self.voice_bank_path = Path(voice_bank_path)
        self.voice_bank_reader = VoiceBankReader(self.voice_bank_path)
        
        # 加载所有模型
        print("正在加载DiffSinger模型...")
        self.dsdur = self.voice_bank_reader.get_dsdur()
        self.dspitch = self.voice_bank_reader.get_dspitch()
        self.dsvariance = self.voice_bank_reader.get_dsvariance()
        self.dsacoustic = self.voice_bank_reader.get_dsacoustic()
        self.dsvocoder = self.voice_bank_reader.get_dsvocoder()
        
        # 创建各个预测器
        self.pred_duration = PredDuration(self.dsdur)
        self.pred_pitch = PredPitch(self.dspitch)
        self.pred_variance = PredVariance(self.dsvariance)
        self.pred_acoustic = PredAcoustic(self.dsacoustic)
        self.pred_vocoder = PredVocoder(self.dsvocoder)
        
        print("所有模型加载完成！")
        
        # 获取可用说话人列表
        self.available_speakers = []
        if self.dsdur.speakers:
            self.available_speakers = [spk.speaker_name for spk in self.dsdur.speakers]
            print(f"可用说话人: {self.available_speakers}")
        else:
            print("没有可用的说话人")
    
    def predict_full_pipeline(self, 
                            ds: DSReader.DSSection,
                            lang: str = 'zh',
                            speaker: Optional[str] = None,
                            key_shift: int = 0,
                            pitch_steps: int = 10,
                            variance_steps: int = 10,
                            acoustic_steps: int = 50,
                            gender: float = 0,
                            output_dir: str = "output",
                            save_intermediate: bool = True) -> Dict[str, Any]:
        """
        执行完整的预测流程
        
        Args:
            ds: DS段落对象（只需要基础信息：text, ph_seq, ph_num, note_midi, note_dur, note_rest）
            lang: 语言代码
            speaker: 说话人名称，如果为None则随机选择
            key_shift: 音高移调（半音数）
            duration_steps: 时长预测采样步数
            pitch_steps: 音高预测采样步数
            variance_steps: 方差预测采样步数
            acoustic_steps: 声学模型采样步数
            gender: 性别参数, [-1, 1], -1表示男性，1表示女性，默认为0
            output_dir: 输出目录
            save_intermediate: 是否保存中间结果
            
        Returns:
            dict: 包含所有预测结果的字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择说话人
        if speaker is None and self.available_speakers:
            speaker = random.choice(self.available_speakers)
            print(f"随机选择说话人: {speaker}")
        elif speaker and speaker not in self.available_speakers:
            print(f"警告: 指定的说话人 '{speaker}' 不存在")
            if self.available_speakers:
                speaker = self.available_speakers[0]
                print(f"使用默认说话人: {speaker}")
            else:
                speaker = None
        
        print(f"\n{'='*80}")
        print(f"开始完整预测流程")
        print(f"{'='*80}")
        print(f"输入信息:")
        print(f"  文本: {ds['text']}")
        print(f"  音素序列: {ds['ph_seq']}")
        print(f"  语言: {lang}")
        print(f"  说话人: {speaker}")
        print(f"  音高移调: {key_shift:+d} 半音")
        
        results = {}
        
        # 第一步：时长预测
        print(f"\n{'-'*60}")
        print("第一步：时长预测")
        print(f"{'-'*60}")
        
        # 创建用于时长预测的DS副本（移除可能存在的时长信息）
        ds_for_duration = DSReader.DSSection(dict(ds))
        # if 'ph_dur' in ds_for_duration:
        #     del ds_for_duration['ph_dur']
        
        ph_dur_pred = self.pred_duration.predict(
            ds_for_duration,
            lang=lang,
            speaker=speaker
        )
        
        # 保存时长预测结果
        if save_intermediate:
            duration_output_path = output_dir / "step1_duration.ds"
            self.pred_duration.save_duration_results(ds_for_duration, ph_dur_pred, duration_output_path)
        
        results['ph_dur'] = ph_dur_pred
        
        # 第二步：音高预测
        print(f"\n{'-'*60}")
        print("第二步：音高预测")
        print(f"{'-'*60}")
        
        # 创建包含时长信息的DS段落
        ds_with_duration = DSReader.DSSection(dict(ds))
        ds_with_duration['ph_dur'] = ' '.join([str(round(dur, 6)) for dur in ph_dur_pred.tolist()])
        
        # 移除可能存在的音高信息
        if 'f0_seq' in ds_with_duration:
            del ds_with_duration['f0_seq']
        if 'f0_timestep' in ds_with_duration:
            del ds_with_duration['f0_timestep']
        
        f0_pred = self.pred_pitch.predict(
            ds_with_duration,
            lang=lang,
            speaker=speaker,
            key_shift=key_shift,
            steps=pitch_steps
        )
        
        # 保存音高预测结果
        if save_intermediate:
            pitch_output_path = output_dir / "step2_pitch.ds"
            self.pred_pitch.save_pitch_results(ds_with_duration, f0_pred, pitch_output_path)
        
        results['f0'] = f0_pred
        
        # 第三步：方差预测
        print(f"\n{'-'*60}")
        print("第三步：方差预测")
        print(f"{'-'*60}")
        
        # 创建包含时长和音高信息的DS段落
        ds_with_pitch = DSReader.DSSection(dict(ds_with_duration))
        ds_with_pitch['f0_seq'] = ' '.join([str(round(f, 4)) for f in f0_pred.tolist()])
        ds_with_pitch['f0_timestep'] = str(self.pred_pitch.timestep)
        
        # 移除可能存在的方差信息
        variance_params = ['breathiness', 'voicing', 'tension', 'energy']
        for v_name in variance_params:
            if v_name in ds_with_pitch:
                del ds_with_pitch[v_name]
            if f'{v_name}_timestep' in ds_with_pitch:
                del ds_with_pitch[f'{v_name}_timestep']
        
        variance_pred = self.pred_variance.predict(
            ds_with_pitch,
            lang=lang,
            speaker=speaker,
            key_shift=0,  # 音高移调已经在pitch步骤中应用
            steps=variance_steps,
            retake_all=True
        )
        
        # 保存方差预测结果
        if save_intermediate:
            variance_output_path = output_dir / "step3_variance.ds"
            self.pred_variance.save_variance_results(ds_with_pitch, variance_pred, variance_output_path)
        
        results['variance'] = variance_pred
        
        # 第四步：声学模型预测
        print(f"\n{'-'*60}")
        print("第四步：声学模型预测")
        print(f"{'-'*60}")
        
        # 创建包含所有信息的DS段落
        ds_complete = DSReader.DSSection(dict(ds_with_pitch))
        for v_name, v_pred in variance_pred.items():
            ds_complete[v_name] = ' '.join([str(round(v, 4)) for v in v_pred.tolist()])
            ds_complete[f'{v_name}_timestep'] = str(self.pred_variance.timestep)
        
        mel_pred = self.pred_acoustic.predict(
            ds_complete,
            lang=lang,
            speaker=speaker,
            steps=acoustic_steps,
            gender=gender
        )
        
        # 保存mel频谱图
        if save_intermediate:            
            # 保存mel和f0数据为JSON
            mel_json_path = output_dir / "step4_mel_data.json"
            save_mel_and_f0_as_json(
                mel_pred[0], f0_pred, str(mel_json_path),
                sample_rate=self.dsacoustic.sample_rate,
                hop_size=self.dsacoustic.hop_size,
                num_mel_bins=self.dsacoustic.num_mel_bins,
                mel_fmin=self.dsacoustic.mel_fmin,
                mel_fmax=self.dsacoustic.mel_fmax
            )
            print(f"Mel数据已保存到: {mel_json_path}")
        
        results['mel'] = mel_pred
        
        # 第五步：声码器推理
        print(f"\n{'-'*60}")
        print("第五步：声码器推理")
        print(f"{'-'*60}")
        
        # 准备F0数据用于声码器
        mel_length = mel_pred.shape[1]
        f0_for_vocoder = resample_align_curve(
            f0_pred,
            original_timestep=self.pred_pitch.timestep,
            target_timestep=self.pred_vocoder.timestep,
            align_length=mel_length
        )
        
        wav_pred = self.pred_vocoder.predict(mel_pred, f0_for_vocoder)
        
        # 保存音频文件
        audio_output_path = output_dir / "step5_final_audio.wav"
        self.pred_vocoder.save_wav(wav_pred, audio_output_path)
        print(f"最终音频已保存到: {audio_output_path}")
        
        results['wav'] = wav_pred
        results['audio_path'] = str(audio_output_path)
        
        # 保存完整的预测结果
        complete_ds_path = output_dir / "complete_prediction.ds"
        with open(complete_ds_path, 'w', encoding='utf-8') as f:
            json.dump([dict(ds_complete)], f, ensure_ascii=False, indent=2)
        print(f"完整DS文件已保存到: {complete_ds_path}")
        
        print(f"\n{'='*80}")
        print("完整预测流程完成！")
        print(f"{'='*80}")
        print(f"生成的文件:")
        if save_intermediate:
            print(f"  1. 时长预测: {output_dir}/step1_duration.ds")
            print(f"  2. 音高预测: {output_dir}/step2_pitch.ds")
            print(f"  3. 方差预测: {output_dir}/step3_variance.ds")
            print(f"  4. Mel数据: {output_dir}/step4_mel_data.json")
        print(f"  5. 最终音频: {audio_output_path}")
        print(f"  6. 完整DS文件: {complete_ds_path}")
        
        # 计算音频统计信息
        audio_length = len(wav_pred) / self.dsvocoder.sample_rate
        print(f"\n音频统计信息:")
        print(f"  音频长度: {audio_length:.2f}秒")
        print(f"  采样率: {self.dsvocoder.sample_rate} Hz")
        print(f"  音素数量: {len(ph_dur_pred)}")
        print(f"  总音素时长: {ph_dur_pred.sum():.2f}秒")
        if (f0_pred > 0).any():
            print(f"  F0范围: {f0_pred[f0_pred > 0].min():.2f} ~ {f0_pred[f0_pred > 0].max():.2f} Hz")
            print(f"  有声帧比例: {(f0_pred > 0).sum() / len(f0_pred):.2%}")
        
        return results


def main():
    """主函数"""
    # 语音库路径
    voice_bank_path = Path("/Users/bc/Music/Singers/Qixuan_v2.0.0_DiffSinger_OpenUtau")
    
    if not voice_bank_path.exists():
        print(f"错误: 语音库路径不存在: {voice_bank_path}")
        print("请确保语音库已正确放置在指定路径")
        return
    
    # 创建完整预测器
    pred_all = PredAll(voice_bank_path)
    try:
        pred_all = PredAll(voice_bank_path)
    except Exception as e:
        print(f"错误: 无法加载语音库: {e}")
        return
    
    # 读取DS文件
    ds_file = "samples/07_春江花月夜.ds"
    if not Path(ds_file).exists():
        print(f"错误: DS文件不存在: {ds_file}")
        print("请确保DS文件存在于指定路径")
        return
    
    try:
        ds_reader = DSReader(ds_file)
        ds_sections = ds_reader.read_ds()
        
        if not ds_sections:
            print(f"错误: 无法读取DS文件: {ds_file}")
            return
        
        # 使用第一个段落进行预测
        ds = ds_sections[0]
        
        print(f"读取DS文件成功: {ds_file}")
        print(f"段落数量: {len(ds_sections)}")
        print(f"使用第一个段落进行预测")
        
    except Exception as e:
        print(f"错误: 读取DS文件失败: {e}")
        return
    
    # 执行完整预测流程
    try:
        results = pred_all.predict_full_pipeline(
            ds=ds,
            lang='zh',
            speaker=None,  # 随机选择说话人
            key_shift=4,   # 不移调
            pitch_steps=10,
            variance_steps=10,
            acoustic_steps=10,
            gender=0.0,
            output_dir="output/pred_all",
            save_intermediate=True
        )
        
        print(f"\n🎉 完整预测流程成功完成！")
        print(f"请检查 output/pred_all/ 目录下的生成文件")
        
    except Exception as e:
        print(f"错误: 预测流程失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()