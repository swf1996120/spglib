import numpy as np
import pytest

import spglib


def test_get_symmetry_dataset_rejects_noninteger_atom_types():
    cell = (np.eye(3), np.array([[0.0, 0.0, 0.0]]), [1.5])
    with pytest.raises(spglib.SpglibError, match="integer species labels"):
        spglib.get_symmetry_dataset(cell)
