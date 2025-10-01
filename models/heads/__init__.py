"""Classification heads."""

from .classification_head import (
    ClassificationHead,
    MultiHeadClassificationHead,
    build_classification_head,
)

__all__ = [
    'ClassificationHead',
    'MultiHeadClassificationHead',
    'build_classification_head',
]
