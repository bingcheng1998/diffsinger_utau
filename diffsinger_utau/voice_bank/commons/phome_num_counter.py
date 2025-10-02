from typing import List
from pypinyin import pinyin, Style
from itertools import groupby
from .utils import pinyin_initials, pinyin_finals
from .voice_bank_reader import format_repr

class Phome:
    def __init__(self, phome_seq: List[str]):
        self.phome_seq = phome_seq

        result = list() # [(0: d), (0: ian), (1: b), ...]
        index = 0
        for i, ph in enumerate(phome_seq):
            result.append((index, ph))
            if ph in pinyin_initials:
                index += 1
            elif i + 1 < len(phome_seq) and phome_seq[i + 1] in ['AP', 'SP']:
                index += 1
            elif ph in ['AP', 'SP'] and i + 1 < len(phome_seq) and phome_seq[i + 1] in pinyin_finals:
                index += 1
            elif ph in pinyin_finals and i + 1 < len(phome_seq) and phome_seq[i + 1] in pinyin_finals:
                index += 1
        self.ph_slur = result
        data = [x[0] for x in result]
        counts = [len(list(g)) for _, g in groupby(data)]
        self.ph_num = counts

    def get_ph_num(self):
        return self.ph_num

    def __repr__(self):
        return format_repr("Phome", phome_seq=self.phome_seq, ph_num=self.ph_num, ph_slur=self.ph_slur)

if __name__ == '__main__':
    import json
    sample2 = '而你在心里面要怎么道别,AP er n i z ai x in l i m ian ian ian SP AP y ao z en m e SP d ao b ie ie ie AP'
    sample1 = '也许已经没有明天,SP y E x v y i j ing AP m ei y ou m ing t ian SP AP'
    text = list(sample1.split(',')[0])
    phome_seq = sample1.split(',')[1].split(' ')    
    print(f'phome_seq: {phome_seq}')
    print(f'initials: {pinyin_initials}')
    phome = Phome(phome_seq)
    phome_num = phome.get_ph_num()
    print(phome_num)
    print(sum(phome_num))
    print(len(phome_seq))

    # 读取 samples 下所有ds文件，校验生成的 phome_num 和ds文件中的 phome_num 是否一致。ds就是json文件
    import os
    for file in os.listdir('samples'):
        if file.endswith('.ds'):
            with open(os.path.join('samples', file), 'r') as f:
                sections = json.load(f)
                for section in sections:
                    phome_num = section['ph_num']
                    phome_seq = section['ph_seq'].split(' ')
                    phome = Phome(phome_seq)
                    result = ' '.join(map(str, phome.get_ph_num()))
                    if phome_num != result:
                        print(f'💗phome_seq: {phome_seq}\n  GT: {phome_num}\nPred: {result}\nfile: {file}')
