'''
这个文件用于读取 ds 文件，并返回一个字典
'''

import sys
import os
import json
from typing import Any, Dict, List
from collections import defaultdict
from pathlib import Path
from pypinyin import pinyin, Style
import pkg_resources

from .voice_bank_reader import format_repr
from .utils import TextDictionary
from .phome_num_counter import Phome


class DSReader:
    
    class DSSection(defaultdict):
        def __init__(self, ds: dict):
            super().__init__(str)
            if ds:                 # 如果传了 dict，就更新
                self.update(ds)
            self.ds_replacer = DSTextReplacer()
            self.verify()

        def gen_ph_num(self) -> list:
            phome = Phome(self.get_list('ph_seq'))
            return phome

        def replace(self, text: str):
            return self.ds_replacer.replace(self, text)

        def get_list(self, key: str) -> list:
            if key not in self:
                return []
            result = self.get(key, '').split(' ')
            if 'seq' in key or 'text' in key:
                return result
            if key in ['note_slur', 'ph_num']:
                return [int(x) for x in result]
            else:
                return [float(x) for x in result]

        def verify(self):
            assert self.get('offset') is not None
            assert self.get('text') is not None
            assert self.get('ph_seq') is not None
            assert self.get('ph_num') is not None
            assert self.get('note_seq') is not None
            assert self.get('note_slur') is not None
            assert self.get('note_dur') is not None
            
            phome_seq_num = len(self.get_list('ph_seq'))
            phome_num_sum = sum(self.get_list('ph_num'))
            assert phome_seq_num == phome_num_sum
            
        def has_dur(self):
            return 'ph_dur' in self
        
        def has_pitch(self):
            return 'f0_seq' in self and 'f0_timestep' in self
        
        def has_breathiness(self):
            return 'breathiness' in self and 'breathiness_timestep' in self
        
        def has_energy(self):
            return 'energy' in self and 'energy_timestep' in self
        
        def has_tension(self):
            return 'tension' in self and 'tension_timestep' in self
        
        def has_voicing(self):
            return 'voicing' in self and 'voicing_timestep' in self
        
        def __repr__(self):
            return format_repr("DSSection",
                             offset=self.get('offset'),
                             text=self.get('text'),
                             ph_seq=self.get('ph_seq'),
                             ph_dur=self.get('ph_dur'),
                             ph_num=self.get('ph_num'),
                             note_seq=self.get('note_seq'),
                             note_slur=self.get('note_slur'),
                             note_dur=self.get('note_dur'),
                             f0_seq=self.get('f0_seq'),
                             f0_timestep=self.get('f0_timestep'),
                             breathiness=self.get('breathiness'),
                             breathiness_timestep=self.get('breathiness_timestep'),
                             energy=self.get('energy'),
                             energy_timestep=self.get('energy_timestep'),
                             tension=self.get('tension'),
                             tension_timestep=self.get('tension_timestep'),
                             voicing=self.get('voicing'),
                             voicing_timestep=self.get('voicing_timestep'))

    def __init__(self, ds_path: Path):
        self.ds_path = ds_path
        self.ds = []

    def read_ds(self) -> List[DSSection]:
        with open(self.ds_path, 'r') as f:
            raw_ds = json.load(f)
            
            # 将原始数据转换为DSSection对象
            self.ds = []
            for section in raw_ds:
                self.ds.append(self.DSSection(dict(section)))
                
        return self.ds

    def __repr__(self):
        return format_repr("DSReader",
                         ds_path=self.ds_path,
                         ds=self.ds[0])


class DSTextReplacer:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DSTextReplacer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.opencpop_dict = self.get_opencpop_dict()
            DSTextReplacer._initialized = True
    
    def get_opencpop_dict(self):
        """使用 pkg_resources 获取字典文件"""
        try:
            # 使用 pkg_resources 获取包内资源
            resource_path = pkg_resources.resource_filename('diffsinger_utau', 'dictionaries/opencpop-extension.txt')
            with open(resource_path, 'r', encoding='utf-8') as f:
                result = {
                    'AP': 'AP',
                    'SP': 'SP'
                }
                for line in f:
                    if '\t' in line:
                        result[line.split("\t")[0].strip()] = line.split("\t")[1].strip()
                return result
        except (FileNotFoundError, ModuleNotFoundError):
            # 降级到相对路径（开发环境）
            fallback_path = os.path.join(os.path.dirname(__file__), '../../dictionaries/opencpop-extension.txt')
            with open(fallback_path, 'r', encoding='utf-8') as f:
                result = {
                    'AP': 'AP',
                    'SP': 'SP'
                }
                for line in f:
                    if '\t' in line:
                        result[line.split("\t")[0].strip()] = line.split("\t")[1].strip()
                return result

    def get_phonemes(self, text, opencpop_dict):
        pinyins = [x[0] for x in pinyin(text, style=Style.NORMAL)]
        result = []
        for py in pinyins:
            py = py.strip()
            if not py:
                continue
            if py in opencpop_dict:
                result.append(opencpop_dict[py])
            else:
                result.append(py)
        result = str(' '.join(result)).split()
        return result

    def replace(self, ds, text: str):
        phonemes = self.get_phonemes(text, self.opencpop_dict)
        ds["ph_seq"] = ' '.join(phonemes)
        ds["ph_num"] = ' '.join(map(str, Phome(phonemes).get_ph_num()))
        return ds

if __name__ == "__main__":
    from pathlib import Path
    
    ds_path = Path('samples/00_我多想说再见啊.ds')
    ds_reader = DSReader(ds_path)
    ds = ds_reader.read_ds()
    print(ds[0])
    print(ds[0].gen_ph_num())