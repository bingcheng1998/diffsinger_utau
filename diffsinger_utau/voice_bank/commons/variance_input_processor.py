#!/usr/bin/env python3

from typing import Union
import json
import numpy as np
from pathlib import Path
import sys
import os
from collections import OrderedDict
import pathlib

import torch
from torch import nn
import librosa
from scipy import interpolate
import torch.nn.functional as F

from .voice_bank_reader import VoiceBankReader
from .tts_modules import LengthRegulator, mel2ph_to_dur

VARIANCE_CHECKLIST = ['energy', 'breathiness', 'voicing', 'tension']

hparams = {
    # 'work_dir': 'data/opencpop/variance_binary_dur_pitch',
    'midi_smooth_width': 0.06,
    # 'dictionaries': {
    #     'zh': 'dictionaries/opencpop-extension.txt'
    # },
    'hop_size': 512,
    'audio_sample_rate': 44100
}

def resample_align_curve(points: np.ndarray, original_timestep: float, target_timestep: float, align_length: int):
    t_max = (len(points) - 1) * original_timestep
    curve_interp = np.interp(
        np.arange(0, t_max, target_timestep),
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)
    delta_l = align_length - len(curve_interp)
    if delta_l < 0:
        curve_interp = curve_interp[:align_length]
    elif delta_l > 0:
        curve_interp = np.concatenate((curve_interp, np.full(delta_l, fill_value=curve_interp[-1])), axis=0)
    return curve_interp

def norm_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf
    return f0

def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0

class VarianceInputProcessor:
    def __init__(
            self, vb_reader: VoiceBankReader.VarianceBase, device=None, 
            predictions: set = set(['dur', 'pitch', 'breathiness', 'voicing', 'tension'])
    ):
        # super().__init__(device=device)
        self.vb_reader = vb_reader  # 保存引用以便后续使用
        self.phoneme_dictionary = vb_reader.phonemes
        print(self.phoneme_dictionary)
        self.lang_map = vb_reader.languages.content if vb_reader.languages is not None else {}
        # lang_map_fn = pathlib.Path(hparams['work_dir']) / 'lang_map.json'
        # if lang_map_fn.exists():
        #     with open(lang_map_fn, 'r', encoding='utf8') as f:
        #         self.lang_map = json.load(f)
        self.lr = LengthRegulator()
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        self.device = 'cpu'
        smooth_kernel_size = round(hparams['midi_smooth_width'] / self.timestep)
        self.smooth = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=smooth_kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        ).eval().to(self.device)
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, smooth_kernel_size).astype(np.float32) * np.pi
        ).to(self.device))
        smooth_kernel /= smooth_kernel.sum()
        self.smooth.weight.data = smooth_kernel[None, None]

        glide_types = hparams.get('glide_types', [])
        assert 'none' not in glide_types, 'Type name \'none\' is reserved and should not appear in glide_types.'
        self.glide_map = {
            'none': 0,
            **{
                typename: idx + 1
                for idx, typename in enumerate(glide_types)
            }
        }

        self.auto_completion_mode = len(predictions) == 0
        self.global_predict_dur = 'dur' in predictions
        self.global_predict_pitch = 'pitch' in predictions
        self.variance_prediction_set = predictions.intersection(VARIANCE_CHECKLIST)
        self.global_predict_variances = len(self.variance_prediction_set) > 0
        
    def preprocess_input(
            self,
            param, idx=0,
            load_dur: bool = False,
            load_pitch: bool = False
    ):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :param load_dur: whether ph_dur is loaded
        :param load_pitch: whether pitch is loaded
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()

        lang = param.get('lang', 'zh')
        if lang is None:
            assert len(self.lang_map) <= 1, (
                "This is a multilingual model. "
                "Please specify a language by --lang option."
            )
        elif self.lang_map:
            assert lang in self.lang_map, f'Unrecognized language name: \'{lang}\'.'
        # 检查模型是否使用语言ID
        use_lang_id = getattr(self.vb_reader, 'use_lang_id', False) if hasattr(self.vb_reader, 'use_lang_id') else hparams.get('use_lang_id', False)
        if use_lang_id:
            languages = torch.LongTensor([[
                (
                    self.lang_map[lang if '/' not in p else p.split('/', maxsplit=1)[0]]
                    if self.phoneme_dictionary.is_cross_lingual(p)
                    else self.lang_map.get(lang, 0)
                )
                for p in param['ph_seq'].split()
            ]]).to(self.device)  # [B=1, T_ph]
            batch['languages'] = languages
        txt_tokens = torch.LongTensor([
            self.phoneme_dictionary.encode(param['ph_seq'], lang=lang)
        ]).to(self.device)  # [B=1, T_ph]
        T_ph = txt_tokens.shape[1]
        batch['tokens'] = txt_tokens
        ph_num = torch.from_numpy(np.array([param['ph_num'].split()], np.int64)).to(self.device)  # [B=1, T_w]
        ph2word = self.lr(ph_num)  # => [B=1, T_ph]
        T_w = int(ph2word.max())
        batch['ph2word'] = ph2word

        note_midi = np.array(
            [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in param['note_seq'].split()],
            dtype=np.float32
        )
        note_rest = note_midi < 0
        if np.all(note_rest):
            # All rests, fill with constants
            note_midi = np.full_like(note_midi, fill_value=60.)
        else:
            # Interpolate rest values
            interp_func = interpolate.interp1d(
                np.where(~note_rest)[0], note_midi[~note_rest],
                kind='nearest', fill_value='extrapolate'
            )
            note_midi[note_rest] = interp_func(np.where(note_rest)[0])
        note_midi = torch.from_numpy(note_midi).to(self.device)[None]  # [B=1, T_n]
        note_rest = torch.from_numpy(note_rest).to(self.device)[None]  # [B=1, T_n]

        T_n = note_midi.shape[1]
        note_dur_sec = torch.from_numpy(np.array([param['note_dur'].split()], np.float32)).to(self.device)  # [B=1, T_n]
        note_acc = torch.round(torch.cumsum(note_dur_sec, dim=1) / self.timestep + 0.5).long()
        note_dur = torch.diff(note_acc, dim=1, prepend=note_acc.new_zeros(1, 1))
        mel2note = self.lr(note_dur)  # [B=1, T_s]
        T_s = mel2note.shape[1]

        summary['words'] = T_w
        summary['notes'] = T_n
        summary['tokens'] = T_ph
        summary['frames'] = T_s
        summary['seconds'] = '%.2f' % (T_s * self.timestep)

        if load_dur:
            # Get mel2ph if ph_dur is needed
            ph_dur_sec = torch.from_numpy(
                np.array([param['ph_dur'].split()], np.float32)
            ).to(self.device)  # [B=1, T_ph]
            ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=1) / self.timestep + 0.5).long()
            ph_dur = torch.diff(ph_acc, dim=1, prepend=ph_acc.new_zeros(1, 1))
            mel2ph = self.lr(ph_dur, txt_tokens == 0)
            if mel2ph.shape[1] != T_s:  # Align phones with notes
                mel2ph = F.pad(mel2ph, [0, T_s - mel2ph.shape[1]], value=mel2ph[0, -1])
                ph_dur = mel2ph_to_dur(mel2ph, T_ph)
            # Get word_dur from ph_dur and ph_num
            word_dur = note_dur.new_zeros(1, T_w + 1).scatter_add(
                1, ph2word, ph_dur
            )[:, 1:]  # => [B=1, T_w]
        else:
            ph_dur = None
            mel2ph = None
            # Get word_dur from note_dur and note_slur
            is_slur = torch.BoolTensor([[int(s) for s in param['note_slur'].split()]]).to(self.device)  # [B=1, T_n]
            note2word = torch.cumsum(~is_slur, dim=1)  # [B=1, T_n]
            word_dur = note_dur.new_zeros(1, T_w + 1).scatter_add(
                1, note2word, note_dur
            )[:, 1:]  # => [B=1, T_w]

        batch['ph_dur'] = ph_dur
        batch['mel2ph'] = mel2ph

        mel2word = self.lr(word_dur)  # [B=1, T_s]
        if mel2word.shape[1] != T_s:  # Align words with notes
            mel2word = F.pad(mel2word, [0, T_s - mel2word.shape[1]], value=mel2word[0, -1])
            word_dur = mel2ph_to_dur(mel2word, T_w)
        batch['word_dur'] = word_dur

        batch['note_midi'] = note_midi
        batch['note_dur'] = note_dur
        batch['note_rest'] = note_rest
        if hparams.get('use_glide_embed', False) and param.get('note_glide') is not None:
            batch['note_glide'] = torch.LongTensor(
                [[self.glide_map.get(x, 0) for x in param['note_glide'].split()]]
            ).to(self.device)
        else:
            batch['note_glide'] = torch.zeros(1, T_n, dtype=torch.long, device=self.device)
        batch['mel2note'] = mel2note

        # Calculate and smoothen the frame-level MIDI pitch, which is a step function curve
        frame_midi_pitch = torch.gather(
            F.pad(note_midi, [1, 0]), 1, mel2note
        )  # => frame-level MIDI pitch, [B=1, T_s]
        base_pitch = self.smooth(frame_midi_pitch)
        batch['base_pitch'] = base_pitch

        if ph_dur is not None:
            # Phone durations are available, calculate phoneme-level MIDI.
            mel2pdur = torch.gather(F.pad(ph_dur, [1, 0], value=1), 1, mel2ph)  # frame-level phone duration
            ph_midi = frame_midi_pitch.new_zeros(1, T_ph + 1).scatter_add(
                1, mel2ph, frame_midi_pitch / mel2pdur
            )[:, 1:]
        else:
            # Phone durations are not available, calculate word-level MIDI instead.
            mel2wdur = torch.gather(F.pad(word_dur, [1, 0], value=1), 1, mel2word)
            w_midi = frame_midi_pitch.new_zeros(1, T_w + 1).scatter_add(
                1, mel2word, frame_midi_pitch / mel2wdur
            )[:, 1:]
            # Convert word-level MIDI to phoneme-level MIDI
            ph_midi = torch.gather(F.pad(w_midi, [1, 0]), 1, ph2word)
        ph_midi = ph_midi.round().long()
        batch['midi'] = ph_midi

        if load_pitch:
            f0 = resample_align_curve(
                np.array(param['f0_seq'].split(), np.float32),
                original_timestep=float(param['f0_timestep']),
                target_timestep=self.timestep,
                align_length=T_s
            )
            batch['pitch'] = torch.from_numpy(
                librosa.hz_to_midi(interp_f0(f0)[0]).astype(np.float32)
            ).to(self.device)[None]

        if self.global_predict_dur:
            if load_dur:
                summary['ph_dur'] = 'manual'
            elif self.auto_completion_mode or self.global_predict_dur:
                summary['ph_dur'] = 'auto'
            else:
                summary['ph_dur'] = 'ignored'

        if self.global_predict_pitch:
            if load_pitch:
                summary['pitch'] = 'manual'
            elif self.auto_completion_mode or self.global_predict_pitch:
                summary['pitch'] = 'auto'

                # Load expressiveness
                expr = param.get('expr', 1.)
                if isinstance(expr, (int, float, bool)):
                    summary['expr'] = f'static({expr:.3f})'
                    batch['expr'] = torch.FloatTensor([expr]).to(self.device)[:, None]  # [B=1, T=1]
                else:
                    summary['expr'] = 'dynamic'
                    expr = resample_align_curve(
                        np.array(expr.split(), np.float32),
                        original_timestep=float(param['expr_timestep']),
                        target_timestep=self.timestep,
                        align_length=T_s
                    )
                    batch['expr'] = torch.from_numpy(expr.astype(np.float32)).to(self.device)[None]

            else:
                summary['pitch'] = 'ignored'

        if self.global_predict_variances:
            for v_name in self.variance_prediction_set:
                if self.auto_completion_mode and param.get(v_name) is None or v_name in self.variance_prediction_set:
                    summary[v_name] = 'auto'
                else:
                    summary[v_name] = 'ignored'

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))
        
        batch['ph_seq'] = param['ph_seq']
        batch['text'] = param['text']

        return batch



if __name__ == "__main__":
    from ds_reader import DSReader
    from typing import List

    # 创建语音库读取器
    voice_bank_path = Path("artifacts/JiangKe_DiffSinger_CE_25.06")
    voice_bank_reader = VoiceBankReader(voice_bank_path)
    
    # 获取所有模型
    dsdur = voice_bank_reader.get_dsdur()
    # 读取DS文件
    ds_path = Path('samples/00_我多想说再见啊.1.ds')
    ds_reader = DSReader(ds_path)
    ds: List[DSReader.DSSection] = ds_reader.read_ds()
    ds0: DSReader.DSSection = ds[0]
    ds0['lang'] = 'zh'
    vp = VarianceInputProcessor(dsdur)
    inputs = vp.preprocess_input(ds0, 0, load_dur=True, load_pitch=True)
    for k, v in inputs.items():
        print(f'{k}: {str(v)[:70]}{"..." if len(str(v)) > 70 else ""}')
