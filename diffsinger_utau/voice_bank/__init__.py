from .pred_all import PredAll
from .pred_duration import PredDuration
from .pred_pitch import PredPitch
from .pred_variance import PredVariance
from .pred_acoustic import PredAcoustic
from .pred_vocoder import PredVocoder

__version__ = "0.1.5.dev"

__classifiers__ = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Topic :: Multimedia :: Sound/Audio :: Speech",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

__all__ = [
    "PredAll",
    "PredDuration",
    "PredPitch",
    "PredVariance",
    "PredAcoustic",
    "PredVocoder",
]

