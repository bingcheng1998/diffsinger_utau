#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音高预测脚本
使用ONNX格式的DiffSinger Pitch模型预测基频(F0)
"""

import json
import sys
from pathlib import Path

import numpy as np
import librosa

from .commons.voice_bank_reader import VoiceBankReader
from .commons.ds_reader import DSReader
from .commons.utils import resample_align_curve, encode_phonemes
from .commons.variance_input_processor import VarianceInputProcessor

class PredPitch:
    """音高预测器"""
    
    def __init__(self, dspitch: VoiceBankReader.DSPitch):
        """
        初始化音高预测器
        
        Args:
            dspitch: DSPitch 对象
        """
        self.dspitch = dspitch
        self.timestep = dspitch.hop_size / dspitch.sample_rate

        self.input_processor = VarianceInputProcessor(dspitch)
        
        print(f"音高模型配置加载完成:")
        print(f"  使用语言ID: {dspitch.use_lang_id}")
        print(f"  说话人数量: {len(dspitch.speakers) if dspitch.speakers else 0}")
    
    def _encode_phonemes(self, ph_seq: str, lang: str = 'zh'):
        """编码音素序列"""
        # 使用utils中的编码函数
        tokens = encode_phonemes(ph_seq, self.dspitch.phonemes.content, lang)
        
        # 语言ID处理
        languages = None
        if self.dspitch.use_lang_id and self.dspitch.languages.content:
            languages = []
            for phone in ph_seq.split():
                if '/' in phone:
                    # 跨语言音素
                    lang_code = phone.split('/', maxsplit=1)[0]
                    lang_id = self.dspitch.languages.content.get(lang_code, 0)
                else:
                    lang_id = self.dspitch.languages.content.get(lang, 0)
                languages.append(lang_id)
            languages = np.array(languages, dtype=np.int64)
        
        return tokens, languages
    
    def _process_durations(self, ds: DSReader.DSSection):
        """处理时长信息"""
        # 处理音素时长
        ph_dur_sec = np.array(ds['ph_dur'].split(), dtype=np.float32)
        ph_acc = np.round(np.cumsum(ph_dur_sec) / self.timestep + 0.5).astype(np.int64)
        ph_dur = np.diff(ph_acc, prepend=0)
        
        # 计算总时长（用于确定T_s）
        T_s = int(ph_acc[-1])  # 使用音素时长的最后一个累积值作为总长度
        
        return {
            'ph_dur': ph_dur,
            'T_s': T_s
        }
    
    def _process_note_info(self, ds: DSReader.DSSection, T_s: int):
        """处理音符信息"""
        # 获取音符MIDI和时长
        note_midi = np.array(ds['note_midi'].split(), dtype=np.float32)
        note_dur_sec = np.array(ds['note_dur'].split(), dtype=np.float32)
        note_rest = np.array(ds['note_rest'].split(), dtype=bool)
        
        # 计算音符累积时长
        note_acc = np.round(np.cumsum(note_dur_sec) / self.timestep + 0.5).astype(np.int64)
        note_dur = np.diff(note_acc, prepend=0)
        
        # 创建mel2note映射
        mel2note = np.zeros(T_s, dtype=np.int64)
        start_idx = 0
        for i, dur in enumerate(note_dur):
            end_idx = min(start_idx + dur, T_s)
            mel2note[start_idx:end_idx] = i + 1  # 1-indexed
            start_idx = end_idx
        
        # 创建基础音高
        base_pitch = np.zeros(T_s, dtype=np.float32)
        start_idx = 0
        for i, (midi, dur, is_rest) in enumerate(zip(note_midi, note_dur, note_rest)):
            end_idx = min(start_idx + dur, T_s)
            if not is_rest and midi > 0:
                base_pitch[start_idx:end_idx] = midi
            start_idx = end_idx
        
        return {
            'note_midi': note_midi,
            'note_dur': note_dur,
            'note_rest': note_rest,
            'mel2note': mel2note,
            'base_pitch': base_pitch
        }
    
    def predict(self, ds: DSReader.DSSection, lang: str = 'zh', speaker: str = None, 
                key_shift: int = 0, steps: int = 10):
        """
        预测音高
        
        Args:
            ds: DS段落对象
            lang: 语言代码
            speaker: 说话人名称
            key_shift: 音高移调（半音数）
            steps: 扩散采样步数
        
        Returns:
            np.ndarray: 预测的F0序列
        """
        # 确保模型已加载
        if not self.dspitch.linguistic_model.session:
            self.dspitch.linguistic_model.load_model()
        if not self.dspitch.pitch_model.session:
            self.dspitch.pitch_model.load_model()
        
        print(f"开始音高预测...")
        print(f"  语言: {lang}")
        print(f"  说话人: {speaker}")
        print(f"  音高移调: {key_shift:+d} 半音")
        print(f"  采样步数: {steps}")
        
        # 使用VarianceInputProcessor来处理输入
        inputs = self.input_processor.preprocess_input(ds, 0, load_dur=True, load_pitch=False)
        
        # 获取必要的数据
        encoder_out = inputs['tokens']  # 这里需要先通过linguistic encoder
        ph_dur = inputs['ph_dur']
        note_midi = inputs['note_midi']
        note_rest = inputs['note_rest'] 
        note_dur = inputs['note_dur']
        base_pitch = inputs['base_pitch']
        T_s = base_pitch.shape[1]
        
        word_dur = inputs['word_dur'].numpy().astype(np.int64)
        ph_num_str = ds['ph_num']  # 这是字符串，如 "2 2 1 2 2 2 2 2 1 2 2 2 1 1"
        word_div = np.array(ph_num_str.split(), dtype=np.int64)
        
        # 运行linguistic encoder
        print(f"  运行linguistic encoder...")
        linguistic_inputs = {
            'tokens': inputs['tokens'].detach().numpy().astype(np.int64),
            'ph_dur': inputs['ph_dur'].detach().numpy().astype(np.int64),
            'word_dur': word_dur,
            'word_div': word_div[None].astype(np.int64)
        }
        if 'languages' in inputs:
            linguistic_inputs['languages'] = inputs['languages'].detach().numpy().astype(np.int64)
        else:
            # 如果没有languages，创建默认的语言ID（全部为0）
            linguistic_inputs['languages'] = np.zeros_like(inputs['tokens'].detach().numpy(), dtype=np.int64)
            
        encoder_outputs = self.dspitch.linguistic_model.predict(linguistic_inputs)
        encoder_out = encoder_outputs[0] if isinstance(encoder_outputs, list) else encoder_outputs
        
        # 准备说话人嵌入
        spk_embed = None
        if self.dspitch.speakers:
            if speaker:
                selected_speaker = next((spk for spk in self.dspitch.speakers if spk.speaker_name == speaker), None)
                if selected_speaker:
                    print(f"  使用指定说话人: {speaker}")
                else:
                    selected_speaker = self.dspitch.speakers[0]
                    print(f"  指定说话人不存在，使用默认说话人: {selected_speaker.speaker_name}")
            else:
                selected_speaker = self.dspitch.speakers[0]
                print(f"  使用默认说话人: {selected_speaker.speaker_name}")
            
            # 扩展为时间序列：[1, T_s, embed_dim]
            spk_embed = np.tile(selected_speaker.get_embed()[None, None, :], (1, T_s, 1)).astype(np.float32)
        
        # 准备pitch predictor输入
        pitch_inputs = {
            'encoder_out': encoder_out,
            'ph_dur': inputs['ph_dur'].detach().numpy().astype(np.int64),
            'note_midi': inputs['note_midi'].detach().numpy().astype(np.float32),
            'note_rest': inputs['note_rest'].detach().numpy().astype(np.bool_),
            'note_dur': inputs['note_dur'].detach().numpy().astype(np.int64),
            'pitch': inputs['base_pitch'].detach().numpy().astype(np.float32),  # 使用base_pitch作为初始pitch
            'expr': np.array([[1.0]], dtype=np.float32),  # 默认表现力为1.0
            'retake': np.ones((1, T_s), dtype=np.bool_),  # 重新预测所有帧
            'steps': np.array(steps, dtype=np.int64)
        }
        
        if spk_embed is not None:
            pitch_inputs['spk_embed'] = spk_embed
        
        # 运行pitch predictor
        print(f"  运行pitch predictor...")
        pitch_outputs = self.dspitch.pitch_model.predict(pitch_inputs)
        
        # 获取预测的F0
        f0_pred = pitch_outputs[0][0] if isinstance(pitch_outputs, list) else pitch_outputs[0]
        
        # 转换为Hz
        f0_pred = librosa.midi_to_hz(f0_pred)
        
        # 应用音高移调
        if key_shift != 0:
            voiced_mask = f0_pred > 0
            f0_pred[voiced_mask] *= (2 ** (key_shift / 12.0))
            print(f"  应用音高移调: {key_shift:+d} 半音")
        
        print(f"音高预测完成:")
        print(f"  F0范围: {f0_pred[f0_pred > 0].min():.2f} ~ {f0_pred[f0_pred > 0].max():.2f} Hz")
        print(f"  有声帧比例: {(f0_pred > 0).sum() / len(f0_pred):.2%}")
        
        return f0_pred
    
    def save_pitch_results(self, ds: DSReader.DSSection, f0_pred: np.ndarray, output_path: str):
        """保存音高预测结果"""
        # 复制原始DS段落
        result = dict(ds)
        
        # 添加预测的F0
        result['f0_seq'] = ' '.join([str(round(f, 4)) for f in f0_pred.tolist()])
        result['f0_timestep'] = str(self.timestep)
        
        # 保存为JSON文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
        
        print(f"音高预测结果已保存到: {output_path}")


def main():
    """主函数"""
    # 创建语音库读取器
    voice_bank_path = Path("artifacts/JiangKe_DiffSinger_CE_25.06")
    voice_bank_reader = VoiceBankReader(voice_bank_path)
    
    # 获取音高模型
    dspitch = voice_bank_reader.get_dspitch()
    
    # 创建预测器
    pred_pitch = PredPitch(dspitch)
    
    # 读取DS文件
    ds_file = "samples/00_我多想说再见啊.1.pred_duration.ds"
    ds_reader = DSReader(ds_file)
    ds_sections = ds_reader.read_ds()
    
    if not ds_sections:
        print(f"错误: 无法读取DS文件: {ds_file}")
        return
    
    # 使用第一个段落进行预测
    ds = ds_sections[0]
    print(f"使用DS段落进行预测:")
    print(f"  文本: {ds['text']}")
    print(f"  音素序列: {ds['ph_seq']}")
    print(f"  音素时长: {ds['ph_dur']}")
    print(f"  音符MIDI: {ds['note_midi']}")

    f0_org = ds.get_list('f0_seq')
    f0_timestep_org = float(ds.get('f0_timestep'))
    ds.pop('f0_seq')
    ds.pop('f0_timestep')
    assert 'f0_seq' not in ds and 'f0_timestep' not in ds
    
    # 随机选择说话人
    import random
    if dspitch.speakers:
        available_speakers = [spk.speaker_name for spk in dspitch.speakers]
        selected_speaker = random.choice(available_speakers)
        print(f"随机选择说话人: {selected_speaker}")
    else:
        selected_speaker = None
        print("没有可用的说话人")
    
    print("\n" + "="*60)
    print("音高预测")
    print("="*60)
    
    # 执行音高预测
    f0_pred = pred_pitch.predict(
        ds, 
        lang='zh', 
        speaker=selected_speaker,
        key_shift=0,
        steps=10
    )
    
    # 保存音高预测结果
    pitch_output_path = "output/predicted_pitch.ds"
    pred_pitch.save_pitch_results(ds, f0_pred, pitch_output_path)
    
    # 可视化F0曲线
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        
        # 设置中文字体支持
        try:
            # 尝试设置中文字体
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            # 如果字体设置失败，使用英文标签
            pass
        
        plt.figure(figsize=(12, 6))
        
        # 绘制预测的F0曲线
        time_axis_pred = np.arange(len(f0_pred)) * pred_pitch.timestep
        voiced_mask_pred = f0_pred > 0
        plt.plot(time_axis_pred[voiced_mask_pred], f0_pred[voiced_mask_pred], 'b-', linewidth=1.5, label='Predicted F0')
        
        # 绘制GT的F0曲线
        f0_org_array = np.array(f0_org, dtype=np.float32)
        f0_timestep_org_float = float(f0_timestep_org)
        time_axis_gt = np.arange(len(f0_org_array)) * f0_timestep_org_float
        voiced_mask_gt = f0_org_array > 0
        plt.plot(time_axis_gt[voiced_mask_gt], f0_org_array[voiced_mask_gt], 'r-', linewidth=1.5, label='Ground Truth F0', alpha=0.7)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title('F0 Curve Comparison - GT vs Predicted')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/f0_comparison_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"F0对比曲线图已保存到: output/f0_comparison_curve.png")
    except ImportError:
        print("matplotlib未安装，跳过F0曲线可视化")
    
    print("\n" + "="*60)
    print("音高预测完成！")
    print("="*60)
    print(f"生成的文件:")
    print(f"  音高预测结果: {pitch_output_path}")
    print(f"  F0曲线图: output/f0_comparison_curve.png")
    print(f"\n预测的F0统计信息:")
    print(f"  总帧数: {len(f0_pred)}")
    print(f"  有声帧数: {(f0_pred > 0).sum()}")
    print(f"  有声帧比例: {(f0_pred > 0).sum() / len(f0_pred):.2%}")
    if (f0_pred > 0).any():
        print(f"  F0范围: {f0_pred[f0_pred > 0].min():.2f} ~ {f0_pred[f0_pred > 0].max():.2f} Hz")
        print(f"  平均F0: {f0_pred[f0_pred > 0].mean():.2f} Hz")


if __name__ == "__main__":
    main()