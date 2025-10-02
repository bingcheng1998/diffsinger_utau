#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声码器预测脚本
使用ONNX格式的声码器将mel频谱图转换为wav音频文件
"""

import json
import os
import pathlib
import sys
from pathlib import Path

import numpy as np

from .commons.voice_bank_reader import VoiceBankReader
from .commons.utils import resample_align_curve


class PredVocoder:
    """声码器预测器"""
    
    def __init__(self, ds_vocoder: VoiceBankReader.DSVocoder):
        """
        初始化声码器预测器
        
        Args:
            ds_vocoder: DSVocoder 对象
        """
        self.ds_vocoder = ds_vocoder
        self.timestep = ds_vocoder.hop_size / ds_vocoder.sample_rate
        
        print(f"声码器配置加载完成:")
        print(f"  名称: {ds_vocoder.name}")
        print(f"  采样率: {ds_vocoder.sample_rate}")
        print(f"  Hop size: {ds_vocoder.hop_size}")
        print(f"  Mel bins: {ds_vocoder.num_mel_bins}")
        print(f"  FFT size: {ds_vocoder.fft_size}")
        print(f"  音调可控: {ds_vocoder.pitch_controllable}")
    
    def _prepare_inputs(self, mel_data, f0_data=None):
        """
        准备声码器输入
        
        Args:
            mel_data: mel频谱图数据，形状为 [T, mel_bins] 或 [1, T, mel_bins]
            f0_data: f0数据，形状为 [T] 或 [1, T]，可选
        
        Returns:
            dict: 声码器输入字典
        """
        # 处理mel数据形状
        if len(mel_data.shape) == 2:
            mel_data = mel_data[None, :]  # [1, T, mel_bins]
        elif len(mel_data.shape) == 3 and mel_data.shape[0] == 1:
            pass  # 已经是正确形状
        else:
            raise ValueError(f"不支持的mel数据形状: {mel_data.shape}")
        
        # 准备输入字典
        inputs = {'mel': mel_data.astype(np.float32)}
        
        # 处理f0数据（如果声码器支持音调控制）
        if f0_data is not None:
            if len(f0_data.shape) == 1:
                f0_data = f0_data[None, :]  # [1, T]
            inputs['f0'] = f0_data.astype(np.float32)
        
        return inputs
    
    def predict(self, mel_data, f0_data=None, device='cpu'):
        """
        使用声码器将mel转换为wav
        
        Args:
            mel_data: mel频谱图数据
            f0_data: f0数据，可选
            device: 推理设备
        
        Returns:
            np.ndarray: wav音频数据
        """
        # 确保模型已加载
        if not self.ds_vocoder.model.session:
            self.ds_vocoder.model.load_model(device)
        
        # 准备输入
        inputs = self._prepare_inputs(mel_data, f0_data)
        
        print(f"开始声码器推理...")
        print(f"  Mel形状: {inputs['mel'].shape}")
        assert 'f0' in inputs or 'f0' not in self.ds_vocoder.model.input_names, f'inputs: {inputs.keys()}, but requires {self.ds_vocoder.model.input_names}'
        print(f"  F0形状: {inputs['f0'].shape}")
        
        # 执行ONNX推理
        wav_data = self.ds_vocoder.model.predict(inputs)
        
        print(f"声码器推理完成，wav形状: {wav_data.shape}")
        
        # 如果wav_data是2D的，取第一个样本
        if len(wav_data.shape) == 2 and wav_data.shape[0] == 1:
            wav_data = wav_data[0]
        
        return wav_data
    
    def load_from_json(self, json_path):
        """
        从JSON文件加载mel和f0数据
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            tuple: (mel_data, f0_data, metadata)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            data = data[0]
        
        mel_data = np.array(data['mel'], dtype=np.float32)
        f0_data = np.array(data['f0'], dtype=np.float32)
        metadata = data.get('metadata', {})
        
        if len(mel_data.shape) == 2:
            # 第一维度增加batch维度
            mel_data = mel_data[np.newaxis, :]
        if mel_data.shape[2] != self.ds_vocoder.num_mel_bins and mel_data.shape[1] == self.ds_vocoder.num_mel_bins:
            # 交换1、2维度
            mel_data = np.swapaxes(mel_data, 1, 2)
        
        # 重采样F0数据到mel的时间分辨率
        if len(f0_data) > 0:
            mel_length = mel_data.shape[1]  # mel的时间帧数
            if len(f0_data) != mel_length:
                # 计算F0的原始时间步长（假设从DS文件提取的F0时间步长为0.005）
                f0_timestep = 0.005
                # 计算mel的时间步长
                mel_timestep = self.timestep
                
                print(f"重采样F0数据: {len(f0_data)} -> {mel_length}")
                print(f"  F0时间步长: {f0_timestep:.4f}s")
                print(f"  Mel时间步长: {mel_timestep:.4f}s")
                
                f0_data = resample_align_curve(
                    f0_data,
                    original_timestep=f0_timestep,
                    target_timestep=mel_timestep,
                    align_length=mel_length
                )
        
        print(f"从JSON文件加载数据: {json_path}")
        print(f"  Mel形状: {mel_data.shape}")
        print(f"  F0形状: {f0_data.shape}")
        print(f"  采样率: {metadata.get('sample_rate', 'unknown')}")
        
        return mel_data, f0_data, metadata
    
    def save_wav(self, wav_data, output_path):
        """
        保存wav文件
        
        Args:
            wav_data: wav音频数据
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 将音频数据转换为16位PCM格式
        if wav_data.dtype != np.float32:
            wav_data = wav_data.astype(np.float32)
        
        # 归一化到[-1, 1]范围
        wav_data = np.clip(wav_data, -1.0, 1.0)
        
        # 转换为16位整数
        wav_int16 = (wav_data * 32767).astype(np.int16)
        
        # 保存为wav文件
        import wave
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(self.ds_vocoder.sample_rate)
            wav_file.writeframes(wav_int16.tobytes())
        
        # 计算音频长度
        audio_length = len(wav_data) / self.ds_vocoder.sample_rate
        
        print(f"Wav文件已保存到: {output_path}")
        print(f"  音频长度: {audio_length:.2f}秒")
        print(f"  采样率: {self.ds_vocoder.sample_rate} Hz")
        print(f"  声道数: 1 (单声道)")


def main():
    """主函数"""
    # 创建语音库读取器
    voice_bank_path = Path("artifacts/JiangKe_DiffSinger_CE_25.06")
    voice_bank_reader = VoiceBankReader(voice_bank_path)
    
    # 获取声码器
    ds_vocoder = voice_bank_reader.get_dsvocoder()
    
    # 创建声码器预测器
    pred_vocoder = PredVocoder(ds_vocoder)
    
    # 从JSON文件加载mel和f0数据
    json_path = "output/predicted_data.json"
    if not Path(json_path).exists():
        print(f"错误: JSON文件不存在: {json_path}")
        print("请先运行 pred_acoustic.py 生成mel数据")
        return
    
    mel_data, f0_data, metadata = pred_vocoder.load_from_json(json_path)
    
    # 执行声码器推理
    print("开始声码器推理...")
    wav_data = pred_vocoder.predict(mel_data, f0_data)
    
    # 保存wav文件
    output_path = "output/predicted_audio.wav"
    pred_vocoder.save_wav(wav_data, output_path)
    
    print("声码器推理完成！")


if __name__ == "__main__":
    main()
