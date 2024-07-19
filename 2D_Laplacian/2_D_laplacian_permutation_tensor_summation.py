from MyQuantum import *
import numpy as np
import numpy_Kron_overload as npk
import matplotlib.pyplot as plt
from LCU.Prep_And_Slect import *

if __name__ == "__main__":
    D = npk.Matrix(np.array([[2, -1, 0, -1],
                                [-1, 2, -1, 0],
                                [0, -1, 2, -1],
                                [-1, 0, -1, 2]]))

    dx = D ** I_4
    dy = I_4 ** D
    TwoD_Laplace = dx + dy


    print(dx.real)
    print(dy.real)
    print(TwoD_Laplace.real)
    plt.imshow(TwoD_Laplace.real)
    plt.colorbar()

    P1 = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0]])
    P2 = P1 @ P1
    P3 = P2 @ P1

    print((2 * I_4 + (-1) * P1 + (-1) * P3))



