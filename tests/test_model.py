import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import predict_crop

def test_predict_crop():
    result = predict_crop(50, 50, 50, 25.0, 60.0, 6.5, 100.0)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
