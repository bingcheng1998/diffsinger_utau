import torch
from torch.nn import functional as F

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000

class RhythmRegulator(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, ph_dur, ph2word, word_dur):
        """
        Example (no batch dim version):
            1. ph_dur = [4,2,3,2]
            2. word_dur = [3,4,2], ph2word = [1,2,2,3]
            3. word_dur_in = [4,5,2]
            4. alpha_w = [0.75,0.8,1], alpha_ph = [0.75,0.8,0.8,1]
            5. ph_dur_out = [3,1.6,2.4,2]
        :param ph_dur: [B, T_ph]
        :param ph2word: [B, T_ph]
        :param word_dur: [B, T_w]
        """
        ph_dur = ph_dur.float() * (ph2word > 0)
        word_dur = word_dur.float()
        word_dur_in = ph_dur.new_zeros(ph_dur.shape[0], ph2word.max() + 1).scatter_add(
            1, ph2word, ph_dur
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        alpha_w = word_dur / word_dur_in.clamp(min=self.eps)  # avoid dividing by zero
        alpha_ph = torch.gather(F.pad(alpha_w, [1, 0]), 1, ph2word)  # [B, T_w] => [B, T_ph]
        ph_dur_out = ph_dur * alpha_ph
        return ph_dur_out.round().long()


class LengthRegulator(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, dur, dur_padding=None, alpha=None):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        """
        assert alpha is None or alpha > 0
        if alpha is not None:
            dur = torch.round(dur.float() * alpha).long()
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.long())
        token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
        dur_cumsum = torch.cumsum(dur, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)

        pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask.long()).sum(1)
        return mel2ph

def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur