import numpy as np
import pytest
import spglib


@pytest.mark.parametrize(
    "numbers",
    [
        [1.0],
        [1.5],
        np.array([1.0]),
        [np.nan],
        [np.inf],
        [True],
        ["1"],
        [1 + 0j],
        np.array([1], dtype=object),
    ],
)
def test_get_symmetry_dataset_rejects_noninteger_atom_types(numbers):
    """Even integral-valued floats must not be cast into integer labels."""
    cell = (np.eye(3), np.array([[0.0, 0.0, 0.0]]), numbers)
    with pytest.raises(spglib.SpglibError, match="array-like of integers"):
        spglib.get_symmetry_dataset(cell)


@pytest.mark.parametrize(
    "numbers",
    [
        [1],
        (1,),
        np.array([1], dtype=np.int16),
        np.array([1], dtype=np.int32),
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.uint8),
        [-1],
    ],
)
def test_get_symmetry_dataset_accepts_integer_atom_types(numbers):
    """Integer array-likes retain the symmetry of a primitive cubic cell."""
    cell = (np.eye(3), np.array([[0.0, 0.0, 0.0]]), numbers)
    dataset = spglib.get_symmetry_dataset(cell)
    assert dataset.number == 221
    np.testing.assert_array_equal(dataset.equivalent_atoms, [0])


@pytest.mark.parametrize("numbers", [1, [[1]], [[1, 2], [3]]])
def test_get_symmetry_dataset_rejects_invalid_atom_type_shapes(numbers):
    """Scalar, multidimensional and ragged inputs are not per-atom labels."""
    cell = (np.eye(3), np.array([[0.0, 0.0, 0.0]]), numbers)
    with pytest.raises(spglib.SpglibError):
        spglib.get_symmetry_dataset(cell)
