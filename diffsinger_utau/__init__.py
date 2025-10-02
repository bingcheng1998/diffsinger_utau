"""
DiffSinger UTAU - 基于 DiffSinger 的 UTAU 推理工具包

这是一个用于 DiffSinger 模型推理的 Python 包，提供了完整的语音合成流水线。
"""

from .voice_bank import (
    PredAll,
    PredDuration,
    PredPitch,
    PredVariance,
    PredAcoustic,
    PredVocoder,
    __version__
)

__all__ = [
    "PredAll",
    "PredDuration", 
    "PredPitch",
    "PredVariance",
    "PredAcoustic",
    "PredVocoder",
    "__version__"
]