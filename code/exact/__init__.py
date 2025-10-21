"""Exact representations of models"""

from signature_vol.exact.linear_operator import (
    SignatureLinearOperator,
    SignatureLinearOperatorBuilder,
    inner,
)
from signature_vol.exact.signature import Signature

__all__ = [
    "SignatureLinearOperator",
    "SignatureLinearOperatorBuilder",
    "Signature",
    "inner",
]
