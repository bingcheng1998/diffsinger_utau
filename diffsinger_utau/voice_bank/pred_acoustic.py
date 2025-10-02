'''
声学模型用于从完整的 ds 文件预测 mel 频谱
'''

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .commons.voice_bank_reader import VoiceBankReader
from .commons.ds_reader import DSReader
from .commons.utils import save_mel_and_f0_as_json, resample_align_curve, encode_phonemes, calculate_durations


class PredAcoustic:
    def __init__(self, ds_acoustic: VoiceBankReader.DSAcoustic):
        self.ds_acoustic = ds_acoustic
        self.timestep = ds_acoustic.hop_size / ds_acoustic.sample_rate
        
        # 打印加载信息
        print(f"加载音素映射: {len(self.ds_acoustic.phonemes.content)} 个音素")
        if self.ds_acoustic.use_lang_id and self.ds_acoustic.languages.content:
            print(f"加载语言映射: {self.ds_acoustic.languages.content}")
        if self.ds_acoustic.speakers:
            speaker_names = [spk.speaker_name for spk in self.ds_acoustic.speakers]
            print(f"加载说话人映射: {speaker_names}")
    
    def _validate_ds_section(self, ds: DSReader.DSSection) -> bool:
        """验证DS段落是否包含所有必需的参数"""
        # 确保模型已加载以获取input_names
        if not self.ds_acoustic.acoustic_model.session:
            self.ds_acoustic.acoustic_model.load_model()
        
        # 基础必需参数（不依赖模型输入）
        basic_required = ['ph_seq', 'ph_dur']
        missing_params = []
        
        # 检查基础参数
        for param in basic_required:
            if param not in ds:
                missing_params.append(param)
        
        # 检查F0参数（如果模型需要）
        if 'f0' in self.ds_acoustic.acoustic_model.input_names:
            if not ds.has_pitch():
                missing_params.extend(['f0_seq', 'f0_timestep'])
        
        # 检查方差参数（如果模型需要）
        variance_params = ['breathiness', 'voicing', 'tension', 'energy']
        for v_name in variance_params:
            if v_name in self.ds_acoustic.acoustic_model.input_names:
                if v_name not in ds or f'{v_name}_timestep' not in ds:
                    missing_params.extend([v_name, f'{v_name}_timestep'])
        
        # 注意：gender 和 velocity 参数是可选的，如果没有提供会使用默认值
        # 所以不需要在验证中检查它们
        
        if missing_params:
            print(f"警告: 缺少必需参数: {missing_params}")
            print(f"模型需要的输入: {self.ds_acoustic.acoustic_model.input_names}")
            return False
        
        return True
    
    
    def _prepare_inputs(self, ds: DSReader.DSSection, lang: str = None, speaker: str = None, gender: float = 0) -> Dict[str, np.ndarray]:
        """准备模型输入"""
        # 验证输入
        if not self._validate_ds_section(ds):
            raise ValueError("DS段落缺少必需参数")
        
        # 语言处理
        lang_map = self.ds_acoustic.languages.content if self.ds_acoustic.languages is not None else {}
        if lang is None and self.ds_acoustic.use_lang_id:
            if len(lang_map) > 1:
                # 优先使用中文，如果没有则使用第一个可用语言
                if 'zh' in lang_map:
                    lang = 'zh'
                elif lang_map:
                    lang = list(lang_map.keys())[0]
        
        # 音素序列处理
        txt_tokens = encode_phonemes(ds['ph_seq'], self.ds_acoustic.phonemes.content, lang)
        
        # 时长处理
        durations, length = calculate_durations(ds['ph_dur'], self.timestep)
        
        # 准备输入字典
        inputs = {
            'tokens': txt_tokens[None],  # 添加batch维度
            'durations': durations[None]  # 添加batch维度
        }
        
        # 语言ID处理 - 根据模型输入动态处理
        if 'languages' in self.ds_acoustic.acoustic_model.input_names and self.ds_acoustic.use_lang_id and lang in lang_map:
            languages = np.array([
                lang_map[lang if '/' not in p else p.split('/', maxsplit=1)[0]]
                if '/' in p else 0
                for p in ds['ph_seq'].split()
            ], dtype=np.int64)
            inputs['languages'] = languages[None]  # 添加batch维度
        
        # F0处理
        f0_seq = resample_align_curve(
            np.array(ds['f0_seq'].split(), dtype=np.float32),
            float(ds['f0_timestep']),
            self.timestep,
            length
        )
        inputs['f0'] = f0_seq[None]  # 添加batch维度
        
        # 方差参数处理 - 根据模型输入动态处理
        variance_params = ['breathiness', 'voicing', 'tension', 'energy']
        for v_name in variance_params:
            # 只处理模型实际需要的方差参数
            if v_name in self.ds_acoustic.acoustic_model.input_names:
                if v_name in ds and f'{v_name}_timestep' in ds:
                    v_seq = resample_align_curve(
                        np.array(ds[v_name].split(), dtype=np.float32),
                        float(ds[f'{v_name}_timestep']),
                        self.timestep,
                        length
                    )
                    inputs[v_name] = v_seq[None]  # 添加batch维度
                else:
                    # 如果没有提供该方差参数，使用默认值
                    inputs[v_name] = np.zeros((1, length), dtype=np.float32)
        
        # 音调偏移处理 - 根据模型输入动态处理
        if 'gender' in self.ds_acoustic.acoustic_model.input_names and self.ds_acoustic.use_key_shift_embed:
            if gender is None:
                gender = ds.get('gender', 0.0)
            assert gender <= 1 and gender >= -1, f"gender should be between -1 and 1, but get {gender}"
            if isinstance(gender, (int, float, bool)):
                shift_min, shift_max = self.ds_acoustic.augmentation_args.get(
                    'random_pitch_shifting', {}).get('range', [-5.0, 5.0])
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                inputs['gender'] = np.full((1, length), key_shift_value, dtype=np.float32)
            else:
                # 动态音调偏移
                gender_seq = resample_align_curve(
                    np.array(gender.split(), dtype=np.float32),
                    float(ds['gender_timestep']),
                    self.timestep,
                    length
                )
                shift_min, shift_max = self.ds_acoustic.augmentation_args.get(
                    'random_pitch_shifting', {}).get('range', [-5.0, 5.0])
                gender_mask = gender_seq >= 0
                key_shift_seq = gender_seq * (gender_mask * shift_max + (1 - gender_mask) * abs(shift_min))
                inputs['gender'] = np.clip(
                    key_shift_seq.astype(np.float32)[None],
                    a_min=shift_min, a_max=shift_max
                )
        
        # 速度处理 - 根据模型输入动态处理
        if 'velocity' in self.ds_acoustic.acoustic_model.input_names and self.ds_acoustic.use_speed_embed:
            velocity = ds.get('velocity')
            if velocity is None:
                inputs['velocity'] = np.full((1, length), 1.0, dtype=np.float32)
            else:
                speed_min, speed_max = self.ds_acoustic.augmentation_args.get(
                    'random_time_stretching', {}).get('range', [0.8, 1.2])
                speed_seq = resample_align_curve(
                    np.array(velocity.split(), dtype=np.float32),
                    float(ds['velocity_timestep']),
                    self.timestep,
                    length
                )
                inputs['velocity'] = np.clip(
                    speed_seq.astype(np.float32)[None],
                    a_min=speed_min, a_max=speed_max
                )
        
        # 添加其他必需的输入 - 根据模型输入动态处理
        if 'steps' in self.ds_acoustic.acoustic_model.input_names:
            inputs['steps'] = np.array(50, dtype=np.int64)
        if 'depth' in self.ds_acoustic.acoustic_model.input_names:
            inputs['depth'] = np.array(self.ds_acoustic.max_depth, dtype=np.float32)
        
        # 添加说话人嵌入（如果模型需要）
        if 'spk_embed' in self.ds_acoustic.acoustic_model.input_names:
            if self.ds_acoustic.speakers:
                # 选择说话人
                if speaker is None:
                    # 如果没有指定说话人，使用第一个可用的
                    selected_speaker = self.ds_acoustic.speakers[0]
                    print(f"使用默认说话人: {selected_speaker.speaker_name}")
                else:
                    # 查找指定的说话人
                    selected_speaker = None
                    for spk in self.ds_acoustic.speakers:
                        if spk.speaker_name == speaker:
                            selected_speaker = spk
                            break
                    
                    if selected_speaker is None:
                        print(f"警告: 说话人 '{speaker}' 不存在，使用默认说话人: {self.ds_acoustic.speakers[0].speaker_name}")
                        selected_speaker = self.ds_acoustic.speakers[0]
                    else:
                        print(f"使用指定说话人: {selected_speaker.speaker_name}")
                
                spk_embed = selected_speaker.get_embed()
                # 扩展说话人嵌入到时间序列
                spk_embed_expanded = np.tile(spk_embed[None, None, :], (1, length, 1))
                inputs['spk_embed'] = spk_embed_expanded.astype(np.float32)
            else:
                # 使用默认的说话人嵌入（零向量）
                embed_size = 256  # 假设嵌入维度为256
                inputs['spk_embed'] = np.zeros((1, length, embed_size), dtype=np.float32)
        
        return inputs
    
    def predict(self, ds: DSReader.DSSection, lang: str = None, speaker: str = None, steps: int = None, gender: float = 0, device: str = 'cpu') -> np.ndarray:
        """预测mel频谱"""
        # 确保模型已加载
        if not self.ds_acoustic.acoustic_model.session:
            self.ds_acoustic.acoustic_model.load_model(device)
        
        # 准备输入
        inputs = self._prepare_inputs(ds, lang, speaker, gender)

        if steps is not None:
            inputs['steps'] = np.array(steps, dtype=np.int64)
        
        # 使用ONNX模型进行推理
        mel_pred = self.ds_acoustic.acoustic_model.predict(inputs)
        
        return mel_pred

    
if __name__ == "__main__":
    import random
    from pathlib import Path
    
    # 加载声库
    voice_bank_path = Path('artifacts/JiangKe_DiffSinger_CE_25.06')
    voice_bank_reader = VoiceBankReader(voice_bank_path)
    pred_acoustic = PredAcoustic(voice_bank_reader.get_dsacoustic())
    
    # 读取DS文件
    ds_path = Path('samples/08_full_prediction.ds')
    ds_reader = DSReader(ds_path)
    ds = ds_reader.read_ds()
    ds0 = ds[0]
    
    # 随机选择说话人
    if pred_acoustic.ds_acoustic.speakers:
        available_speakers = [spk.speaker_name for spk in pred_acoustic.ds_acoustic.speakers]
        selected_speaker = random.choice(available_speakers)
        print(f"随机选择说话人: {selected_speaker}")
    else:
        selected_speaker = None
        print("没有可用的说话人")
    
    # 进行推理
    print("开始推理...")
    mel = pred_acoustic.predict(ds0, speaker=selected_speaker)
    print(f"推理完成，mel形状: {mel.shape}")
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # 保存mel和f0数据为JSON
    f0_data = resample_align_curve(
        np.array(ds0['f0_seq'].split(), dtype=np.float32),
        original_timestep=float(ds0['f0_timestep']),
        target_timestep=pred_acoustic.timestep,
        align_length=mel.shape[1]
    )
    json_data_path = output_dir / 'predicted_data.json'
    save_mel_and_f0_as_json(
        mel, f0_data, str(json_data_path),
        pred_acoustic.ds_acoustic.sample_rate,
        pred_acoustic.ds_acoustic.hop_size,
        pred_acoustic.ds_acoustic.num_mel_bins,
        pred_acoustic.ds_acoustic.mel_fmin,
        pred_acoustic.ds_acoustic.mel_fmax
    )
    
    # 导入其他预测器
    from .pred_vocoder import PredVocoder
    dsvocoder = voice_bank_reader.get_dsvocoder()
    pred_vocoder = PredVocoder(dsvocoder)
    pred_vocoder = PredVocoder(dsvocoder)
    
    wav_data = pred_vocoder.predict(mel, f0_data)
    
    # 保存音频文件
    pred_vocoder.save_wav(wav_data, output_dir / "predicted_audio.wav")
    
    
    print("推理完成！")