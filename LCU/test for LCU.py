import Prep_And_Slect as ps
import numpy as np

A = np.array([[1, 2, 1, 0, 0],
              [0, 1, 2, 1, 0],
              [0, 0, 1, 2, 1],
              [1, 0, 0, 1, 2],
              [2, 1, 0, 0, 1]], dtype=complex)
def test(A):
    """
    test the block encoding for a banded and cyclic matrix by LCU
    """
    Result = ps.LCU(A)
    return Result