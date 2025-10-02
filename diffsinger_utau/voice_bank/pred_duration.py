#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时长预测脚本
使用ONNX格式的DiffSinger Duration模型预测音素时长
"""

import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Union

from .commons.voice_bank_reader import VoiceBankReader
from .commons.ds_reader import DSReader
from .commons.variance_input_processor import VarianceInputProcessor
from .commons.tts_modules import RhythmRegulator


class PredDuration:
    """时长预测器"""
    
    def __init__(self, dsdur: VoiceBankReader.DSDur):
        """
        初始化时长预测器
        
        Args:
            dsdur: DSDur 对象
        """
        self.dsdur = dsdur
        # 从配置中计算正确的时间步长
        hop_size = dsdur.config.get('hop_size', 512)
        sample_rate = dsdur.config.get('sample_rate', 44100)
        self.timestep = hop_size / sample_rate  # hop_size / sample_rate
        
        print(f"时长模型配置加载完成:")
        print(f"  使用语言ID: {dsdur.use_lang_id}")
        print(f"  说话人数量: {len(dsdur.speakers) if dsdur.speakers else 0}")
        
        # 初始化输入处理器
        self.input_processor = VarianceInputProcessor(dsdur)
        
        # 初始化节奏调节器
        self.rhythm_regulator = RhythmRegulator()
    
    def _prepare_linguistic_inputs(self, inputs, ds):
        """准备linguistic encoder输入"""
        # 从输入中提取需要的数据
        tokens = inputs['tokens'].numpy().astype(np.int64)
        word_dur = inputs['word_dur'].numpy().astype(np.int64)
        
        # 直接从DS段落中获取 ph_num (word_div)
        # ph_num 表示每个词包含的音素数量
        ph_num_str = ds['ph_num']  # 这是字符串，如 "2 2 1 2 2 2 2 2 1 2 2 2 1 1"
        word_div = np.array(ph_num_str.split(), dtype=np.int64)
        
        # 准备linguistic encoder输入
        linguistic_inputs = {
            'tokens': tokens,
            'word_div': word_div[None].astype(np.int64),  # 添加batch维度
            'word_dur': word_dur,
        }
        
        # 添加语言信息（如果需要）
        if self.dsdur.use_lang_id:
            if 'languages' in inputs:
                languages = inputs['languages'].numpy().astype(np.int64)
                linguistic_inputs['languages'] = languages
            else:
                # 如果没有语言信息，创建默认的语言ID（中文=3）
                T_ph = tokens.shape[1]
                languages = np.full((1, T_ph), 3, dtype=np.int64)  # 默认中文
                linguistic_inputs['languages'] = languages
        
        return linguistic_inputs
    
    def _prepare_duration_inputs(self, encoder_out, x_masks, inputs, spk_embed=None):
        """准备duration predictor输入"""
        duration_inputs = {
            'encoder_out': encoder_out,
            'x_masks': x_masks,
            'ph_midi': inputs['midi'].numpy().astype(np.int64),
        }
        
        # 添加说话人嵌入
        if spk_embed is not None:
            duration_inputs['spk_embed'] = spk_embed
        
        return duration_inputs
    

    
    def predict(self, ds: DSReader.DSSection, lang: str = 'zh', speaker: Optional[str] = None):
        """
        预测音素时长
        
        Args:
            ds: DS段落对象
            lang: 语言代码
            speaker: 说话人名称
        
        Returns:
            np.ndarray: 预测的音素时长（秒）
        """
        # 确保模型已加载
        if not self.dsdur.linguistic_model.session:
            self.dsdur.linguistic_model.load_model()
        if not self.dsdur.dur_model.session:
            self.dsdur.dur_model.load_model()
        
        print(f"开始时长预测...")
        print(f"  语言: {lang}")
        print(f"  说话人: {speaker}")
        
        # 使用输入处理器处理输入数据
        inputs = self.input_processor.preprocess_input(ds, 0, load_dur=False, load_pitch=False)
        
        # 准备linguistic encoder输入
        linguistic_inputs = self._prepare_linguistic_inputs(inputs, ds)
        
        print(f"  linguistic输入形状:")
        for k, v in linguistic_inputs.items():
            print(f"    {k}: {v.shape}")
        
        # 运行linguistic encoder
        print(f"  运行linguistic encoder...")
        encoder_outputs = self.dsdur.linguistic_model.predict(linguistic_inputs)
        
        # 解析encoder输出
        if isinstance(encoder_outputs, list):
            encoder_out = encoder_outputs[0]  # encoder_out
            x_masks = encoder_outputs[1] if len(encoder_outputs) > 1 else None  # x_masks
        else:
            encoder_out = encoder_outputs
            x_masks = None
        
        print(f"  encoder输出形状: {encoder_out.shape}")
        if x_masks is not None:
            print(f"  x_masks形状: {x_masks.shape}")
        
        # 准备说话人嵌入
        spk_embed = None
        if self.dsdur.speakers:
            # 选择说话人
            if speaker:
                selected_speaker = next((spk for spk in self.dsdur.speakers if spk.speaker_name == speaker), None)
                if selected_speaker:
                    print(f"  使用指定说话人: {speaker}")
                else:
                    selected_speaker = self.dsdur.speakers[0]
                    print(f"  指定说话人不存在，使用默认说话人: {selected_speaker.speaker_name}")
            else:
                selected_speaker = self.dsdur.speakers[0]
                print(f"  使用默认说话人: {selected_speaker.speaker_name}")
            
            # 说话人嵌入：[1, 1, embed_dim] - 需要3维张量
            spk_embed = selected_speaker.get_embed()[None, None].astype(np.float32)
            print(f"  说话人嵌入形状: {spk_embed.shape}")
        
        # 准备duration predictor输入
        duration_inputs = self._prepare_duration_inputs(encoder_out, x_masks, inputs, spk_embed)
        
        print(f"  duration输入形状:")
        for k, v in duration_inputs.items():
            print(f"    {k}: {v.shape}")
        
        # 运行duration predictor
        print(f"  运行duration predictor...")
        duration_outputs = self.dsdur.dur_model.predict(duration_inputs)
        
        # 后处理输出
        if isinstance(duration_outputs, list):
            ph_dur_pred = duration_outputs[0]
        else:
            ph_dur_pred = duration_outputs
        
        # 应用RhythmRegulator进行后处理
        ph_dur_pred_tensor = torch.from_numpy(ph_dur_pred).float()
        ph2word = inputs['ph2word']
        word_dur = inputs['word_dur']
        
        # 使用commons中的RhythmRegulator
        ph_dur_regulated = self.rhythm_regulator(ph_dur_pred_tensor, ph2word, word_dur)
        
        # 移除batch维度并转换为秒
        ph_dur_regulated = ph_dur_regulated[0].numpy()  # 移除batch维度
        ph_dur_sec = ph_dur_regulated * self.timestep
        
        print(f"时长预测完成:")
        print(f"  预测时长数量: {len(ph_dur_sec)}")
        print(f"  时长范围: {ph_dur_sec.min():.3f}s ~ {ph_dur_sec.max():.3f}s")
        print(f"  总时长: {ph_dur_sec.sum():.3f}s")
        
        return ph_dur_sec
    
    def save_duration_results(self, ds: DSReader.DSSection, ph_dur_pred: np.ndarray, output_path: Union[str, Path]):
        """保存时长预测结果"""
        # 复制原始DS段落
        result = dict(ds)
        
        # 添加预测的时长参数
        result['ph_dur'] = ' '.join([str(round(dur, 6)) for dur in ph_dur_pred.tolist()])
        
        # 保存为JSON文件
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump([result], f, ensure_ascii=False, indent=2)
        
        print(f"时长预测结果已保存到: {output_path_obj}")

def cumulate_ph_dur_by_ph_num(ph_dur: list, ph_num: list)-> list:
    """根据音素数量累加时长"""
    ph_num = copy.deepcopy(ph_num)
    result = [0] * len(ph_num)
    dur_idx = 0
    num_idx = 0
    assert sum(ph_num) == len(ph_dur), f"音素数量与时长数量不一致 {sum(ph_num)} vs {len(ph_dur)}"
    while dur_idx < len(ph_dur) and num_idx < len(ph_num):
        if ph_num[num_idx] > 0:
            result[num_idx] += ph_dur[dur_idx]
            ph_num[num_idx] -= 1
            dur_idx += 1
        else:
            num_idx += 1

    return result


def main():
    """主函数"""
    # 创建语音库读取器
    voice_bank_path = Path("artifacts/JiangKe_DiffSinger_CE_25.06")
    voice_bank_reader = VoiceBankReader(voice_bank_path)
    
    # 获取时长模型
    dsdur = voice_bank_reader.get_dsdur()
    
    # 创建预测器
    pred_duration = PredDuration(dsdur)
    
    # 读取DS文件
    ds_file = "samples/00_我多想说再见啊.1.ds"
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
    print(f"  音素数量: {ds['ph_num']}")
    org_ph_dur = ds.get_list('ph_dur')
    ds.pop('ph_dur')
    assert 'ph_dur' not in ds
    
    # 随机选择说话人
    import random
    if dsdur.speakers:
        available_speakers = [spk.speaker_name for spk in dsdur.speakers]
        selected_speaker = random.choice(available_speakers)
        print(f"随机选择说话人: {selected_speaker}")
    else:
        selected_speaker = None
        print("没有可用的说话人")
    
    print("\n" + "="*60)
    print("时长预测")
    print("="*60)
    
    # 执行时长预测
    ph_dur_pred = pred_duration.predict(
        ds, 
        lang='zh', 
        speaker=selected_speaker if selected_speaker else None
    )
    
    # 保存时长预测结果
    duration_output_path = "samples/00_我多想说再见啊.1.pred_duration.ds"
    pred_duration.save_duration_results(ds, ph_dur_pred, duration_output_path)
    
    # 显示预测结果
    print(f"\n预测的音素时长:")
    ph_seq = ds['ph_seq'].split()
    for i, (ph, dur) in enumerate(zip(ph_seq, ph_dur_pred)):
        print(f"  {i+1:2d}. {ph:8s} -> {dur:.3f}s")
    
    print(f"\n时长预测完成！")

    org_cum = cumulate_ph_dur_by_ph_num(org_ph_dur, ds.get_list('ph_num'))
    pred_cum = cumulate_ph_dur_by_ph_num(ph_dur_pred, ds.get_list('ph_num'))
    cum_diff = np.array(org_cum) - np.array(pred_cum)
    print(f"原始时长累加: {org_cum}")
    print(f"预测时长累加: {pred_cum}")
    print(f"时长累加差异: {cum_diff}")

    assert np.allclose(org_cum, pred_cum, 0.01), f"时长累加差异过大: {cum_diff}"

    print(f"生成的文件: {duration_output_path}")


if __name__ == "__main__":
    main()