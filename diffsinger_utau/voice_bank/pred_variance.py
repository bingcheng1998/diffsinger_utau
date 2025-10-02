#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方差预测脚本
使用ONNX格式的DiffSinger Variance模型预测breathiness、voicing、tension等声音表现参数
"""

from typing import Union
import json
import sys
from pathlib import Path

import numpy as np
import librosa

from .commons.voice_bank_reader import VoiceBankReader
from .commons.ds_reader import DSReader
from .commons.utils import resample_align_curve, encode_phonemes
from .commons.utils import save_mel_and_f0_as_json
from .commons.variance_input_processor import VarianceInputProcessor

class PredVariance:
    """方差预测器"""
    
    def __init__(self, dsvariance: VoiceBankReader.DSVariance):
        """
        初始化方差预测器
        
        Args:
            dsvariance: DSVariance 对象
        """
        self.dsvariance = dsvariance
        self.timestep = dsvariance.hop_size / dsvariance.sample_rate
        
        print(f"方差模型配置加载完成:")
        print(f"  预测breathiness: {dsvariance.predict_breathiness}")
        print(f"  预测voicing: {dsvariance.predict_voicing}")
        print(f"  预测tension: {dsvariance.predict_tension}")
        print(f"  预测energy: {dsvariance.predict_energy}")
        print(f"  使用语言ID: {dsvariance.use_lang_id}")
        print(f"  说话人数量: {len(dsvariance.speakers) if dsvariance.speakers else 0}")
        
        # 构建预测参数列表
        self.variance_prediction_list = []
        if dsvariance.predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        if dsvariance.predict_voicing:
            self.variance_prediction_list.append('voicing')
        if dsvariance.predict_tension:
            self.variance_prediction_list.append('tension')
        if dsvariance.predict_energy:
            self.variance_prediction_list.append('energy')
            
        if len(self.variance_prediction_list) == 0:
            for v_name in ['breathiness', 'voicing', 'tension', 'energy']:
                if v_name in self.dsvariance.variance_model.input_names:
                    self.variance_prediction_list.append(v_name)
        
        print(f"  预测参数列表: {self.variance_prediction_list}")
        
        # 初始化输入处理器
        self.input_processor = VarianceInputProcessor(dsvariance, predictions=set(self.variance_prediction_list))
    

    
    def _prepare_variance_inputs(self, encoder_out, ph_dur, pitch, current_variances, 
                                retake_mask, spk_embed=None, steps=10):
        """准备variance predictor输入"""
        variance_inputs = {
            'encoder_out': encoder_out,
            'ph_dur': ph_dur,
            'pitch': pitch,
            'retake': retake_mask,
            'steps': np.array(steps, dtype=np.int64)
        }
        
        # 添加说话人嵌入
        if spk_embed is not None:
            variance_inputs['spk_embed'] = spk_embed
        
        # 添加当前的variance参数（用于retake机制）
        for v_name in ['breathiness', 'voicing', 'tension', 'energy']:
            if v_name in current_variances:
                variance_inputs[v_name] = current_variances[v_name]
            else:
                # 如果没有该参数，使用零值
                variance_inputs[v_name] = np.zeros_like(pitch, dtype=np.float32)
        
        return variance_inputs
    
    def predict(self, ds: DSReader.DSSection, lang: str = 'zh', speaker: Union[str, None]= None, 
                key_shift: int = 0, steps: int = 10, retake_all: bool = True):
        """
        预测方差参数
        
        Args:
            ds: DS段落对象
            lang: 语言代码
            speaker: 说话人名称
            key_shift: 音高移调（半音数）
            steps: 扩散采样步数
            retake_all: 是否重新预测所有参数
        
        Returns:
            dict: 预测的方差参数
        """
        # 确保模型已加载
        if not self.dsvariance.linguistic_model.session:
            print(f"加载linguistic模型...")
            self.dsvariance.linguistic_model.load_model()
        if not self.dsvariance.variance_model.session:
            print(f"加载variance模型...")
            self.dsvariance.variance_model.load_model()
        
        print(f"开始方差预测...")
        print(f"  语言: {lang}")
        print(f"  说话人: {speaker}")
        print(f"  音高移调: {key_shift:+d} 半音")
        print(f"  采样步数: {steps}")
        
        # 使用输入处理器处理输入数据
        inputs = self.input_processor.preprocess_input(ds, 0, load_dur=True, load_pitch=True)
        
        # 应用音高移调
        if key_shift != 0:
            pitch_midi = inputs['pitch'].detach().numpy()[0]  # 移除batch维度
            voiced_mask = pitch_midi > 0
            pitch_midi[voiced_mask] += key_shift  # MIDI音高直接加减半音数
            inputs['pitch'] = inputs['pitch'].clone()
            inputs['pitch'][0] = torch.from_numpy(pitch_midi)
            print(f"  应用音高移调: {key_shift:+d} 半音")
            
        ph_num_str = ds['ph_num']  # 这是字符串，如 "2 2 1 2 2 2 2 2 1 2 2 2 1 1"
        word_div = np.array(ph_num_str.split(), dtype=np.int64)
        word_dur = inputs['word_dur'].numpy().astype(np.int64)
        
        # 准备linguistic encoder输入
        linguistic_inputs = {
            'tokens': inputs['tokens'].detach().numpy().astype(np.int64),
            'ph_dur': inputs['ph_dur'].detach().numpy().astype(np.int64),
            'word_div': word_div[None].astype(np.int64),
            'word_dur': word_dur
        }
        
        # 检查模型是否需要languages输入
        if self.dsvariance.use_lang_id:
            assert 'languages' in inputs
            linguistic_inputs['languages'] = inputs['languages'].detach().numpy().astype(np.int64)

        
        # 运行linguistic encoder
        print(f"  运行linguistic encoder...")
        encoder_outputs = self.dsvariance.linguistic_model.predict(linguistic_inputs)
        encoder_out = encoder_outputs[0] if isinstance(encoder_outputs, list) else encoder_outputs
        
        # 获取时间序列长度
        T_s = inputs['pitch'].shape[1]
        
        # 准备现有的variance参数（用于retake机制）
        current_variances = {}
        for v_name in self.variance_prediction_list:
            if v_name in ds and not retake_all:
                # 如果输入中已有该参数且不重新预测，重采样到正确长度
                org_data = np.array(ds[v_name].split(), dtype=np.float32)
                v_data = resample_align_curve(
                    org_data,
                    original_timestep=float(ds[f'{v_name}_timestep']),
                    target_timestep=self.timestep,
                    align_length=T_s
                )
                print(f"  重采样 {v_name}: {len(org_data)} -> {len(v_data)}")
                current_variances[v_name] = v_data[None]  # 添加batch维度
            else:
                # 如果没有该参数或需要重新预测，使用零值
                current_variances[v_name] = np.zeros((1, T_s), dtype=np.float32)
        
        # 创建retake mask
        retake_mask = np.zeros((1, T_s, len(self.variance_prediction_list)), dtype=np.bool_)
        
        if retake_all:
            # 重新预测所有参数
            for i, v_name in enumerate(self.variance_prediction_list):
                retake_mask[0, :, i] = True
        else:
            # 只预测没有提供的参数
            for i, v_name in enumerate(self.variance_prediction_list):
                if v_name not in ds:
                    retake_mask[0, :, i] = True
        
        # 准备说话人嵌入
        spk_embed = None
        if self.dsvariance.speakers:
            # 选择说话人
            if speaker:
                selected_speaker = next((spk for spk in self.dsvariance.speakers if spk.speaker_name == speaker), None)
                if selected_speaker:
                    print(f"  使用指定说话人: {speaker}")
                else:
                    selected_speaker = self.dsvariance.speakers[0]
                    print(f"  指定说话人不存在，使用默认说话人: {selected_speaker.speaker_name}")
            else:
                selected_speaker = self.dsvariance.speakers[0]
                print(f"  使用默认说话人: {selected_speaker.speaker_name}")
            
            # 扩展为时间序列：[1, T_s, embed_dim]
            spk_embed = np.tile(selected_speaker.get_embed()[None, None, :], (1, T_s, 1)).astype(np.float32)
        
        # 准备variance predictor输入
        variance_inputs = self._prepare_variance_inputs(
            encoder_out, 
            inputs['ph_dur'].detach().numpy().astype(np.int64),
            inputs['pitch'].detach().numpy().astype(np.float32),
            current_variances,
            retake_mask,
            spk_embed,
            steps
        )
        
        # 运行variance predictor
        print(f"  运行variance predictor...")
        variance_outputs = self.dsvariance.variance_model.predict(variance_inputs)
        
        # 后处理输出
        variance_pred = {}
        for i, v_name in enumerate(self.variance_prediction_list):
            if i < len(variance_outputs):
                variance_pred[v_name] = variance_outputs[i][0]  # 移除batch维度
        
        print(f"方差预测完成:")
        for v_name, v_data in variance_pred.items():
            print(f"  {v_name}: 范围 {v_data.min():.3f} ~ {v_data.max():.3f}")
        
        return variance_pred
    
    def save_variance_results(self, ds: DSReader.DSSection, variance_pred: dict, output_path: str):
        """保存方差预测结果"""
        # 复制原始DS段落
        result = dict(ds)
        
        # 添加预测的方差参数
        for v_name, v_pred in variance_pred.items():
            result[v_name] = ' '.join([str(round(v, 4)) for v in v_pred.tolist()])
            result[f'{v_name}_timestep'] = str(self.timestep)
        
        # 保存为JSON文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
        
        print(f"方差预测结果已保存到: {output_path}")


def main():
    """主函数"""
    # 创建语音库读取器
    voice_bank_path = Path("artifacts/JiangKe_DiffSinger_CE_25.06")
    voice_bank_reader = VoiceBankReader(voice_bank_path)
    
    # 获取所有模型
    dsvariance = voice_bank_reader.get_dsvariance()
    dsacoustic = voice_bank_reader.get_dsacoustic()
    dsvocoder = voice_bank_reader.get_dsvocoder()
    
    # 创建预测器
    pred_variance = PredVariance(dsvariance)
    
    # 导入其他预测器
    from .pred_acoustic import PredAcoustic
    from .pred_vocoder import PredVocoder
    
    pred_acoustic = PredAcoustic(dsacoustic)
    pred_vocoder = PredVocoder(dsvocoder)
    
    # 读取DS文件
    ds_file = "samples/00_我多想说再见啊.ds"
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
    
    # 随机选择说话人
    import random
    if dsvariance.speakers:
        available_speakers = [spk.speaker_name for spk in dsvariance.speakers]
        selected_speaker = random.choice(available_speakers)
        print(f"随机选择说话人: {selected_speaker}")
    else:
        selected_speaker = None
        print("没有可用的说话人")
    
    print("\n" + "="*60)
    print("第一步：方差预测")
    print("="*60)
    
    # 执行方差预测
    variance_pred = pred_variance.predict(
        ds, 
        lang='zh', 
        speaker=selected_speaker,
        key_shift=0,
        steps=10,
        retake_all=True
    )
    
    # 保存方差预测结果
    variance_output_path = "output/predicted_variance.ds"
    pred_variance.save_variance_results(ds, variance_pred, variance_output_path)
    
    print("\n" + "="*60)
    print("第二步：声学模型预测")
    print("="*60)
    
    # 创建包含方差参数的DS段落
    ds_with_variance = DSReader.DSSection(dict(ds))
    for v_name, v_pred in variance_pred.items():
        ds_with_variance[v_name] = ' '.join([str(round(v, 4)) for v in v_pred.tolist()])
        ds_with_variance[f'{v_name}_timestep'] = str(pred_variance.timestep)
    
    # 执行声学模型预测
    mel_pred = pred_acoustic.predict(ds_with_variance, speaker=selected_speaker)
    
    # 准备F0数据用于保存
    f0_data = None
    if ds.has_pitch():
        f0_data = resample_align_curve(
            np.array(ds['f0_seq'].split(), dtype=np.float32),
            original_timestep=float(ds['f0_timestep']),
            target_timestep=pred_acoustic.timestep,
            align_length=mel_pred.shape[1]
        )
    
    save_mel_and_f0_as_json(
        mel_pred[0], f0_data, "output/predicted_data_with_variance.json",
        sample_rate=dsacoustic.sample_rate,
        hop_size=dsacoustic.hop_size,
        num_mel_bins=dsacoustic.num_mel_bins,
        mel_fmin=dsacoustic.mel_fmin,
        mel_fmax=dsacoustic.mel_fmax
    )
    
    print("\n" + "="*60)
    print("第三步：声码器推理")
    print("="*60)
    
    # 执行声码器推理
    wav_data = pred_vocoder.predict(mel_pred, f0_data)
    
    # 保存音频文件
    pred_vocoder.save_wav(wav_data, "output/predicted_audio_with_variance.wav")
    
    print("\n" + "="*60)
    print("完整推理流程完成！")
    print("="*60)
    print(f"生成的文件:")
    print(f"  方差预测结果: {variance_output_path}")
    print(f"  Mel频谱图: output/predicted_mel_with_variance.png")
    print(f"  Mel数据: output/predicted_data_with_variance.json")
    print(f"  音频文件: output/predicted_audio_with_variance.wav")
    print(f"\n请检查生成的音频文件以验证方差预测的效果！")


if __name__ == "__main__":
    main()
