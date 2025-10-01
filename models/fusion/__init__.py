"""Fusion strategies for multi-modal learning."""

from .early import EarlyFusion, EarlyFusionModel
from .late import LateFusion, LateFusionModel
from .meta import METAFusion, METAFusionModel
from .factory import create_fusion_model

__all__ = [
    'EarlyFusion',
    'EarlyFusionModel',
    'LateFusion',
    'LateFusionModel',
    'METAFusion',
    'METAFusionModel',
    'create_fusion_model',
]
