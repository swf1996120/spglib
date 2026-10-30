import numpy as np
import pytest
import spglib


def test_get_symmetry_dataset_rejects_invalid_symprec():
    cell = (np.eye(3), np.array([[0.0, 0.0, 0.0]]), [1])
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="symprec"):
            spglib.get_symmetry_dataset(cell, symprec=value)
