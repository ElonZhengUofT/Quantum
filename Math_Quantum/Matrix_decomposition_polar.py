import sympy as sp
import numpy as np
import scipy.linalg as la
import time
import matplotlib.pyplot  as plt

if __name__ == "__main__":
    a, b, c, d, k = sp.symbols('a b c d k')
    A = sp.Matrix([[a,b,0,b],
                         [b,a,k * b,0],
                         [0,k * b,a, b],
                         [b,0,b,a]])
    dict = {a: 1, b: 2, c: 3, d: 4, k: 3}
    A_n = np.array(A.subs(dict)).astype(np.float64)
    # Polar decomposition
    U, P = la.polar(A_n)
    print(np.round(U, 4))
    print(np.round(P, 4))
    plt.imshow(U)
    plt.colorbar()
    plt.show()