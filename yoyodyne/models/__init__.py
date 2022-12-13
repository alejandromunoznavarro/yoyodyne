"""Model classes and lookup function."""

import pytorch_lightning as pl

from .. import util
from .lstm import LSTMEncoderDecoder, LSTMEncoderDecoderAttention
from .pointer_generator import (
    PointerGeneratorLSTMEncoderDecoderFeatures,
    PointerGeneratorLSTMEncoderDecoderNoFeatures,
)
from .transducer import TransducerFeatures, TransducerNoFeatures
from .transformer import (
    FeatureInvariantTransformerEncoderDecoder,
    TransformerEncoderDecoder,
)


def get_model_cls(
    arch: str, attn: str, include_features: bool
) -> pl.LightningModule:
    """Model factory.

    Args:
        arch (str).
        attn (str).
        include_features (bool).

    Raises:
        NotImplementedError.

    Returns:
        pl.LightningModule.
    """
    model_fac = {
        # fmt: off
        "feature_invariant_transformer":
            FeatureInvariantTransformerEncoderDecoder,
        "pointer_generator_lstm":
            PointerGeneratorLSTMEncoderDecoderFeatures
            if include_features
            else PointerGeneratorLSTMEncoderDecoderNoFeatures,
        # fmt: on
        "transducer": TransducerFeatures
        if include_features
        else TransducerNoFeatures,
        "transformer": TransformerEncoderDecoder,
        "lstm": LSTMEncoderDecoderAttention if attn else LSTMEncoderDecoder,
    }
    try:
        return model_fac[arch]
    except KeyError:
        raise NotImplementedError(f"Architecture {arch} not found")
