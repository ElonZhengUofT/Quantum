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
    test_A = A / np.linalg.norm(A)
    theta_0 = 0
    theta_1 = 0
    prep = ps.prep(theta_0, theta_1)
    select = ps.select()
    Result = (np.kron(ps.identity, np.linalg.inv(prep))) @ select @ (np.kron(ps.identity, prep))