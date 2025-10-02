# -*- coding: utf-8 -*-
'''
此文件是一个类，用于读取声库文件夹中的文件，并提供方法：

1. get_dsdur
2. get_dspitch
3. get_dsvariance
4. get_dsacoustic
5. get_dsvocoder


'''

from typing import DefaultDict
import yaml
import json
import os
from pprint import pformat
import numpy as np

def format_repr(class_name, **kwargs):
    """格式化 __repr__ 输出，使其更美观"""
    lines = [f"<{class_name}>"]
    for key, value in kwargs.items():
        if hasattr(value, '__repr__') and not isinstance(value, (str, int, float, bool, type(None))):
            # 对于有 __repr__ 方法的对象（如 OnnxReader, JsonReader 等），递归格式化
            value_str = str(value)
            if '\n' in value_str:
                # 多行输出，需要调整缩进
                indented_value = "\n".join(f"    {line}" for line in value_str.split("\n"))
                lines.append(f"  {key}:\n{indented_value}")
            else:
                lines.append(f"  {key}: {value}")
        elif isinstance(value, (list, dict)) and len(str(value)) > 50:
            # 对于长列表或字典，使用 pprint 格式化
            formatted_value = pformat(value, width=60, depth=2)
            # 为每行添加适当的缩进
            indented_value = "\n".join(f"    {line}" for line in formatted_value.split("\n"))
            lines.append(f"  {key}:\n{indented_value}")
        else:
            # 对于超长字符串，截断到100个字符
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            lines.append(f"  {key}: {value_str}")
    return "\n".join(lines)

class OnnxReader:
    def __init__(self, onnx_path, preload_models=False):
        self.onnx_path = onnx_path
        self.session = None
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        self.model_size = 0

        if preload_models:
            self.load_model()
        
    def load_model(self, device='cpu'):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            
            # 设置日志等级为 ERROR（只显示错误，不显示 Warning/Info）
            so = ort.SessionOptions()
            so.log_severity_level = 3   # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
            
            # 设置执行提供者
            providers = ['CPUExecutionProvider']
            if device in ['cuda', None] and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif device in ['mps', None] and 'CoreMLExecutionProvider' in ort.get_available_providers():
                providers = [
                    ('CoreMLExecutionProvider', {
                        "ModelFormat": "MLProgram", "MLComputeUnits": "ALL", 
                        "RequireStaticInputShapes": "0", "EnableOnSubgraphs": "0"
                    }),
                ]
            
            # 加载模型
            self.session = ort.InferenceSession(str(self.onnx_path), sess_options=so, providers=providers)
            
            # 获取输入输出信息
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            # 获取输入输出形状
            for inp in self.session.get_inputs():
                self.input_shapes[inp.name] = inp.shape
            for out in self.session.get_outputs():
                self.output_shapes[out.name] = out.shape
            
            # 获取模型文件大小
            import os
            self.model_size = os.path.getsize(self.onnx_path)
            
            print(f"ONNX模型加载成功: {self.onnx_path}")
            print(f"  输入: {self.input_names}")
            print(f"  输出: {self.output_names}")
            print(f"  模型大小: {self.model_size / 1024 / 1024:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"加载ONNX模型失败: {e}")
            return False
    
    def predict(self, inputs):
        """使用ONNX模型进行推理"""
        if self.session is None:
            if not self.load_model():
                raise RuntimeError("模型加载失败")
            
        # 删除 inputs 中无用key
        filtered_inputs = {}
        for name in self.input_names:
            if name in inputs:
                filtered_inputs[name] = inputs[name]
                continue
            if name == 'speedup':
                filtered_inputs['speedup'] = np.array(1, dtype=np.int64)
            elif name == 'energy':
                filtered_inputs[name] = np.array([[0.0]], dtype=np.float32)
            elif name == 'voicing':
                filtered_inputs[name] = np.array([[0.0]], dtype=np.float32)
            elif name == 'breathiness':
                filtered_inputs[name] = np.array([[0.0]], dtype=np.float32)
            elif name == 'tension':
                filtered_inputs[name] = np.array([[0.0]], dtype=np.float32)
        
        try:
            # 运行推理
            outputs = self.session.run(self.output_names, filtered_inputs)
            
            # 如果只有一个输出，直接返回
            if len(outputs) == 1:
                return outputs[0]
            else:
                # 返回所有输出
                return outputs
                
        except Exception as e:
            print(f"ONNX推理失败: {e}")
            raise
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'input_names': self.input_names,
            'output_names': self.output_names,
            'input_shapes': self.input_shapes,
            'output_shapes': self.output_shapes,
            'model_size': self.model_size
        }
    
    def __repr__(self):
        return format_repr("OnnxReader", 
                            onnx_path=self.onnx_path,
                            loaded=self.session is not None,
                            input_names=self.input_names,
                            output_names=self.output_names,
                            model_size_mb=f"{self.model_size / 1024 / 1024:.2f} MB" if self.model_size > 0 else "Unknown")
            
class JsonReader:
    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.content = json.load(f)
            self._phone_to_id = self.content
            langs = set()
            for k in self.content.keys():
                langs.add(k.split('/')[0])
            self._multi_langs = len(langs) > 1
    
    def __repr__(self):
        return format_repr("JsonReader", 
                            json_path=self.json_path,
                            content_length=len(self.content),
                            content_preview=str(self.content)[:100] + ('...' if len(str(self.content)) > 100 else ''))

    def is_cross_lingual(self, phone):
        return False

    def encode_one(self, phone, lang=None):
        if phone in ['AP', 'EP', 'SP', 'GS']:
            return self._phone_to_id[phone]
        if '/' in phone:
            lang, phone = phone.split('/', maxsplit=1)
        if lang is None or not self._multi_langs or phone in self._phone_to_id:
            return self._phone_to_id[phone]
        if '/' not in phone:
            phone = f'{lang}/{phone}'
        return self._phone_to_id[phone]

    def encode(self, sentence, lang=None):
        phones = sentence.strip().split() if isinstance(sentence, str) else sentence
        return [self.encode_one(phone, lang=lang) for phone in phones]
        
class TextReader(JsonReader):
    def __init__(self, json_path):
            self.json_path = json_path
            with open(json_path, 'r') as f:
                self.content = {}
                for i, line in enumerate(f):
                    self.content[line.strip()] = i
                self._phone_to_id = self.content
                langs = set()
                for k in self.content.keys():
                    langs.add(k.split('/')[0])
                self._multi_langs = len(langs) > 1

class SpkEmbededReader:
    '''
    读取说话人嵌入文件，文件是float32的bytes，一共256维
    '''
    def __init__(self, speaker_name, spk_embeded_path):
        self.spk_embeded_path = spk_embeded_path
        self.speaker_name = speaker_name
        with open(spk_embeded_path, 'rb') as f:
            self.content = f.read()
            self.content = np.frombuffer(self.content, dtype=np.float32)
            
    def get_embed(self):
        return self.content
    
    def __repr__(self):
        return format_repr("SpkEmbededReader",
                           speaker_name=self.speaker_name,
                           spk_embeded_path=self.spk_embeded_path,
                           content_length=len(self.content),
                           content=f'{self.content[:5]}...')
            

class VoiceBankReader:
    def __init__(self, voice_bank_path, preload_models=True):
        self.voice_bank_path = voice_bank_path
        self.preload_models = preload_models
        self.dsdur = self.DSDur(voice_bank_path / 'dsdur' / 'dsconfig.yaml', preload_models)
        self.dspitch = self.DSPitch(voice_bank_path / 'dspitch' / 'dsconfig.yaml', preload_models)
        self.dsvariance = self.DSVariance(voice_bank_path / 'dsvariance' / 'dsconfig.yaml', preload_models)
        self.dsacoustic = self.DSAcoustic(voice_bank_path / 'dsconfig.yaml', preload_models)
        self.dsvocoder = self.DSVocoder(voice_bank_path / 'dsvocoder' / 'vocoder.yaml', preload_models)
        self.character = self.CharacterReader(voice_bank_path / 'character.yaml', voice_bank_path / 'character.txt')
        
    '''
    角色读取器
    '''
    class CharacterReader:
        class SubBank:
            def __init__(self, color, prefix, suffix, tone_ranges):
                self.color = color
                self.prefix = prefix
                self.suffix = suffix
                self.tone_ranges = tone_ranges
                
            def __repr__(self):
                return f"SubBank(color={self.color}, prefix={self.prefix}, suffix={self.suffix}, tone_ranges={self.tone_ranges})"
        
        def __init__(self, character_yaml_path, character_txt_path):
            self.character_path = character_yaml_path
            if character_yaml_path and os.path.exists(character_yaml_path):
                with open(character_yaml_path, 'r') as f:
                    self.character: dict = yaml.safe_load(f)
                    self.subbanks = [self.SubBank(**subbank) for subbank in self.character.get('subbanks', [])]
                    self.portrait = self.character.get('portrait')
                    self.portrait_opacity = self.character['portrait_opacity']
                    self.singer_type = self.character['singer_type']
                    self.text_file_encoding = self.character.get('text_file_encoding', 'utf-8')
            
            if character_txt_path is not None and os.path.exists(character_txt_path):
                with open(character_txt_path, 'r') as f:
                    self.character_txt = DefaultDict(str, {line.strip().split('=')[0]: line.strip().split('=')[1] for line in f.readlines() if '=' in line})
                    self.name = self.character_txt['name']
                    self.image = self.character_txt['image']
                    self.author = self.character_txt['author']
                    self.voice = self.character_txt['voice']
                    self.version = self.character_txt['version']
            else:
                self.character_txt = DefaultDict(str)
            
                
        def __repr__(self):
            return format_repr("CharacterReader",
                             character_path=self.character_path,
                             subbanks=self.subbanks,
                             portrait=self.portrait,
                             portrait_opacity=self.portrait_opacity,
                             singer_type=self.singer_type,
                             text_file_encoding=self.text_file_encoding,
                             name=self.name,
                             image=self.image,
                             author=self.author,
                             voice=self.voice,
                             version=self.version)
    '''
    variance 基类
    '''
    class VarianceBase:
        def __init__(self, config_path, preload_models=False):
            self.config_path = config_path
            self.config_dir = config_path.parent
            self.preload_models = preload_models
            self.languages = None
            self.phonemes = None
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self.linguistic_model = OnnxReader(self.config_dir / self.config['linguistic'], preload_models)
                self.use_lang_id = self.config.get('use_lang_id', False)
                self.speakers = [SpkEmbededReader(speaker, self.config_dir / f"{speaker}.emb") for speaker in self.config.get('speakers', [])]
                if 'languages' in self.config:
                    self.languages = JsonReader(self.config_dir / self.config['languages'])
                if 'phonemes' in self.config:
                    if self.config['phonemes'].endswith('json'):
                        self.phonemes = JsonReader(self.config_dir / self.config['phonemes'])
                    elif self.config['phonemes'].endswith('txt'):
                        self.phonemes = TextReader(self.config_dir / self.config['phonemes'])

            
    '''
    duration 时长预测器
    '''
    class DSDur(VarianceBase):
        def __init__(self, config_path, preload_models=False):
            super().__init__(config_path, preload_models)
            self.dur_model = OnnxReader(self.config_dir / self.config['dur'], preload_models)
                    
        def __repr__(self):
            return format_repr("DSDur",
                             config_path=self.config_path,
                             linguistic_model=self.linguistic_model,
                             dur_model=self.dur_model,
                             use_lang_id=self.use_lang_id,
                             speakers=self.speakers,
                             languages=self.languages,
                             phonemes=self.phonemes)

    '''
    pitch 音高预测器
    '''
    class DSPitch(VarianceBase):
        def __init__(self, config_path, preload_models=False):
            super().__init__(config_path, preload_models)
            self.pitch_model = OnnxReader(self.config_dir / self.config['pitch'], preload_models)
            self.hidden_size = self.config.get('hidden_size')
            self.hop_size = self.config['hop_size']
            self.sample_rate = self.config['sample_rate']
            self.use_continuous_acceleration = self.config.get('use_continuous_acceleration')
            self.use_expr = self.config.get('use_expr', False)
            self.use_note_rest = self.config.get('use_note_rest', False)
            self.predict_breathiness = self.config.get('predict_breathiness', False)
            self.predict_dur = self.config.get('predict_dur', False)
            self.predict_energy = self.config.get('predict_energy', False)
            self.predict_tension = self.config.get('predict_tension', False)
            self.predict_voicing = self.config.get('predict_voicing', False)
                
        def __repr__(self):
            return format_repr("DSPitch",
                             config_path=self.config_path,
                             linguistic_model=self.linguistic_model,
                             pitch_model=self.pitch_model,
                             hidden_size=self.hidden_size,
                             hop_size=self.hop_size,
                             sample_rate=self.sample_rate,
                             speakers=self.speakers,
                             use_lang_id=self.use_lang_id,
                             use_continuous_acceleration=self.use_continuous_acceleration,
                             use_expr=self.use_expr,
                             use_note_rest=self.use_note_rest,
                             predict_breathiness=self.predict_breathiness,
                             predict_dur=self.predict_dur,
                             predict_energy=self.predict_energy,
                             predict_tension=self.predict_tension,
                             predict_voicing=self.predict_voicing,
                             languages=self.languages,
                             phonemes=self.phonemes)

    '''
    variance 预测器
    - energy 能量预测器
    - tension 紧张度预测器
    - voicing 声音预测器
    - breathiness 呼吸声预测器
    '''       
    class DSVariance(VarianceBase):
        def __init__(self, config_path, preload_models=False):
            super().__init__(config_path, preload_models)
            self.variance_model = OnnxReader(self.config_dir / self.config['variance'], preload_models)
            self.hidden_size = self.config.get('hidden_size')
            self.hop_size = self.config['hop_size']
            self.sample_rate = self.config['sample_rate']
            self.use_continuous_acceleration = self.config.get('use_continuous_acceleration')
            self.predict_breathiness = self.config.get('predict_breathiness', False)
            self.predict_dur = self.config.get('predict_dur', False)
            self.predict_energy = self.config.get('predict_energy', False)
            self.predict_tension = self.config.get('predict_tension', False)
            self.predict_voicing = self.config.get('predict_voicing', False)
            
                
        def __repr__(self):
            return format_repr("DSVariance",
                             config_path=self.config_path,
                             linguistic_model=self.linguistic_model,
                             variance_model=self.variance_model,
                             hidden_size=self.hidden_size,
                             hop_size=self.hop_size,
                             sample_rate=self.sample_rate,
                             speakers=self.speakers,
                             use_lang_id=self.use_lang_id,
                             use_continuous_acceleration=self.use_continuous_acceleration,
                             predict_breathiness=self.predict_breathiness,
                             predict_dur=self.predict_dur,
                             predict_energy=self.predict_energy,
                             predict_tension=self.predict_tension,
                             predict_voicing=self.predict_voicing,
                             languages=self.languages,
                             phonemes=self.phonemes)
                
    '''
    mel 频谱生成器
    '''
    class DSAcoustic:
        def __init__(self, config_path, preload_models=False):
            self.config_path = config_path
            self.config_dir = config_path.parent
            self.preload_models = preload_models
            self.languages = None
            self.phonemes = None
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self.acoustic_model = OnnxReader(self.config_dir / self.config['acoustic'], preload_models)
                self.fft_size = self.config.get('fft_size')
                self.hidden_size = self.config.get('hidden_size')
                self.hop_size = self.config.get('hop_size', 512)
                self.win_size = self.config.get('win_size')
                self.sample_rate = self.config.get('sample_rate', 44100)
                self.speakers = [SpkEmbededReader(speaker, self.config_dir / f"{speaker}.emb") for speaker in self.config.get('speakers', [])]
                self.use_lang_id = self.config.get('use_lang_id')
                self.use_breathiness_embed = self.config.get('use_breathiness_embed', False)
                self.use_continuous_acceleration = self.config.get('use_continuous_acceleration', False)
                self.use_energy_embed = self.config.get('use_energy_embed', False)
                self.use_key_shift_embed = self.config.get('use_key_shift_embed', False)
                self.use_speed_embed = self.config.get('use_speed_embed', False)
                self.use_tension_embed = self.config.get('use_tension_embed', False)
                self.use_variable_depth = self.config.get('use_variable_depth', False)
                self.use_voicing_embed = self.config.get('use_voicing_embed', False)
                self.vocoder = self.config.get('vocoder', '')
                self.mel_base = self.config.get('mel_base', 'e')
                self.mel_fmax = self.config.get('mel_fmax', 16000)
                self.mel_fmin = self.config.get('mel_fmin', 40)
                self.mel_scale = self.config.get('mel_scale', 'slaney')
                self.num_mel_bins = self.config.get('num_mel_bins', 128)
                self.max_depth = self.config.get('max_depth', 0.6)
                self.augmentation_args = self.config.get('augmentation_args', {})
                if 'languages' in self.config:
                    self.languages = JsonReader(self.config_dir / self.config['languages'])
                if 'phonemes' in self.config:
                    if self.config['phonemes'].endswith('json'):
                        self.phonemes = JsonReader(self.config_dir / self.config['phonemes'])
                    elif self.config['phonemes'].endswith('txt'):
                        self.phonemes = TextReader(self.config_dir / self.config['phonemes'])
                
        def __repr__(self):
            return format_repr("DSAcoustic",
                             config_path=self.config_path,
                             acoustic_model=self.acoustic_model,
                             fft_size=self.fft_size,
                             hidden_size=self.hidden_size,
                             hop_size=self.hop_size,
                             win_size=self.win_size,
                             sample_rate=self.sample_rate,
                             speakers=self.speakers,
                             use_lang_id=self.use_lang_id,
                             use_breathiness_embed=self.use_breathiness_embed,
                             use_continuous_acceleration=self.use_continuous_acceleration,
                             use_energy_embed=self.use_energy_embed,
                             use_key_shift_embed=self.use_key_shift_embed,
                             use_speed_embed=self.use_speed_embed,
                             use_tension_embed=self.use_tension_embed,
                             use_variable_depth=self.use_variable_depth,
                             use_voicing_embed=self.use_voicing_embed,
                             vocoder=self.vocoder,
                             mel_base=self.mel_base,
                             mel_fmax=self.mel_fmax,
                             mel_fmin=self.mel_fmin,
                             mel_scale=self.mel_scale,
                             num_mel_bins=self.num_mel_bins,
                             max_depth=self.max_depth,
                             augmentation_args=self.augmentation_args,
                             languages=self.languages,
                             phonemes=self.phonemes)

    '''
    vocoder 频谱转音频转换器
    '''    
    class DSVocoder:
        def __init__(self, config_path, preload_models=False):
            self.config_path = config_path
            self.config_dir = config_path.parent
            self.preload_models = preload_models
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self.name = self.config['name']
                self.model = OnnxReader(self.config_dir / self.config['model'], preload_models)
                self.sample_rate = self.config['sample_rate']
                self.hop_size = self.config['hop_size']
                self.win_size = self.config['win_size']
                self.fft_size = self.config['fft_size']
                self.num_mel_bins = self.config['num_mel_bins']
                self.mel_fmin = self.config['mel_fmin']
                self.mel_fmax = self.config['mel_fmax']
                self.mel_base = self.config['mel_base']
                self.mel_scale = self.config['mel_scale']
                self.pitch_controllable = self.config.get('pitch_controllable', False)
                self.force_on_cpu = self.config.get('force_on_cpu', False)
                
        def __repr__(self):
            return format_repr("DSVocoder",
                             config_path=self.config_path,
                             name=self.name,
                             model=self.model,
                             sample_rate=self.sample_rate,
                             hop_size=self.hop_size,
                             win_size=self.win_size,
                             fft_size=self.fft_size,
                             num_mel_bins=self.num_mel_bins,
                             mel_fmin=self.mel_fmin,
                             mel_fmax=self.mel_fmax,
                             mel_base=self.mel_base,
                             mel_scale=self.mel_scale,
                             pitch_controllable=self.pitch_controllable,
                             force_on_cpu=self.force_on_cpu)

    def get_dsdur(self):
        """获取 DSDur 模块"""
        return self.dsdur
    
    def get_dspitch(self):
        """获取 DSPitch 模块"""
        return self.dspitch
    
    def get_dsvariance(self):
        """获取 DSVariance 模块"""
        return self.dsvariance
    
    def get_dsacoustic(self):
        """获取 DSAcoustic 模块"""
        return self.dsacoustic
    
    def get_dsvocoder(self):
        """获取 DSVocoder 模块"""
        return self.dsvocoder

        
if __name__ == '__main__':
    import os
    from pathlib import Path
    
    voice_bank_path = Path('artifacts/JiangKe_DiffSinger_CE_25.06')
    voice_bank_reader = VoiceBankReader(voice_bank_path)
    print('=' * 40)
    print(voice_bank_reader.dsdur)
    print('=' * 40)
    print(voice_bank_reader.dspitch)
    print('=' * 40)
    print(voice_bank_reader.dsvariance)
    print('=' * 40)
    print(voice_bank_reader.dsacoustic)
    print('=' * 40)
    print(voice_bank_reader.dsvocoder)
    print('=' * 40)
    print(voice_bank_reader.character)