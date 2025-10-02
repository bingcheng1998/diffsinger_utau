'''
工具函数模块
包含mel频谱保存、数据转换等通用功能
'''

import json
import numpy as np

pinyin_initials = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w',
    'r', 'q', 'z', 'f', 'x', 'd', 'k', 's', 'p', 'y', 'j', 'b', 'h', 'n', 't', 'w', 'ch', 'zh', 'm', 'sh', 'l', 'c', 'g'
]

pinyin_finals = [
    'ao', 'ei', 'ou', 'an', 'en', 'En', 'in', 'un', 'uan', 'vn', 'van', 'ang', 'eng', 'ing', 'ong', 'er', 'ia', 'ua', 'ie', 've', 'iao', 'uei', 'iou', 'ian', 'uen', 'iang', 'uang', 'iong',
    'en', 'ing', 'e', 'ie', 'ui', 'ei', 'eng', 'ian', 'En', 'ai', 'uo', 'er', 'an', 'ua', 'ang', 'ir', 'o', 'iao', 'ia', 'in', 'iong', 'i0', 'vn', 'uang', 'uai', 'uan', 'ou', 'iu', 'van', 'u', 'E', 'i', 'un', 'ong', 'iang', 'a', 've', 'ao', 'v'
]

class TextDictionary(dict):
    def __init__(self, txt_path: str):
        self.txt_path = txt_path
        self.txt_dict = {}
        self.phomes = set()
        self.initials = set()
        self.finals = set()
        self.load_txt()
        super().__init__(self.txt_dict)

    def load_txt(self):
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    k, v = line.split('\t')
                    k = k.strip()
                    v = v.strip()
                    self.txt_dict[k] = v

                    print(k, '=>', v, '=>', v.split())

                    if k != v:
                        self.phomes.add(v.split()[0])
                        self.phomes.add(v.split()[1])
                        self.initials.add(v.split()[0])
                        self.finals.add(v.split()[1])
                    else:
                        self.phomes.add(v)

def save_mel_and_f0_as_json(mel_data: np.ndarray, f0_data: np.ndarray, save_path: str,
                           sample_rate: int, hop_size: int, num_mel_bins: int,
                           mel_fmin: int, mel_fmax: int):
    """将mel和f0数据保存为JSON格式"""
    # 确保数据是numpy数组
    if not isinstance(mel_data, np.ndarray):
        mel_data = np.array(mel_data)
    if not isinstance(f0_data, np.ndarray):
        f0_data = np.array(f0_data)
    
    # 移除batch维度
    if len(mel_data.shape) == 3 and mel_data.shape[0] == 1:
        mel_data = mel_data[0]  # [mel_bins, time_frames]
    if len(f0_data.shape) == 2 and f0_data.shape[0] == 1:
        f0_data = f0_data[0]  # [time_frames]
    
    # 转换为列表以便JSON序列化
    data = {
        'mel': mel_data.tolist(),
        'f0': f0_data.tolist(),
        'metadata': {
            'mel_shape': list(mel_data.shape),
            'f0_shape': list(f0_data.shape),
            'sample_rate': sample_rate,
            'hop_size': hop_size,
            'num_mel_bins': num_mel_bins,
            'mel_fmin': mel_fmin,
            'mel_fmax': mel_fmax
        }
    }
    
    # 保存为JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Mel和F0数据已保存到: {save_path}")
    print(f"  Mel形状: {mel_data.shape}")
    print(f"  F0形状: {f0_data.shape}")


def resample_align_curve(curve: np.ndarray, original_timestep: float, 
                        target_timestep: float, align_length: int) -> np.ndarray:
    """重采样对齐曲线"""
    if len(curve) == 0:
        return np.zeros(align_length, dtype=np.float32)
    
    # 计算原始时间点
    original_times = np.arange(len(curve)) * original_timestep
    # 计算目标时间点
    target_times = np.arange(align_length) * target_timestep
    
    # 线性插值
    aligned_curve = np.interp(target_times, original_times, curve)
    return aligned_curve.astype(np.float32)


def encode_phonemes(ph_seq: str, phoneme_to_id: dict, lang: str = None) -> np.ndarray:
    """编码音素序列"""
    phones = ph_seq.strip().split() if isinstance(ph_seq, str) else ph_seq
    encoded = []
    for phone in phones:
        # 首先尝试直接匹配
        if phone in phoneme_to_id:
            encoded.append(phoneme_to_id[phone])
        # 然后尝试带语言前缀的匹配
        elif lang and f'{lang}/{phone}' in phoneme_to_id:
            encoded.append(phoneme_to_id[f'{lang}/{phone}'])
        else:
            # 如果都找不到，使用0（PAD）
            encoded.append(0)
            print(f"警告: 音素 '{phone}' 未找到，使用PAD")
    return np.array(encoded, dtype=np.int64)


def calculate_durations(ph_dur: str, timestep: float) -> tuple:
    """计算时长和mel2ph"""
    ph_dur_array = np.array(ph_dur.split(), dtype=np.float32)
    ph_acc = np.round(np.cumsum(ph_dur_array) / timestep + 0.5).astype(np.int64)
    durations = np.diff(ph_acc, prepend=0)
    return durations, ph_acc[-1]

if __name__ == '__main__':
    d = TextDictionary('dictionaries/opencpop-extension.txt')
    print(d.initials)
    print(d.finals)