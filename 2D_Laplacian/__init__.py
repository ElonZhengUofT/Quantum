from MyQuantum import *
import numpy as np
import numpy_Kron_overload as npk
import matplotlib.pyplot as plt

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
    # Check the Pj kron I + I kron Pj, try to implement this to implement the 2D Laplace operator.

    o = np.exp(np.pi * 1j / 4)

    Fourier_16 = npk.Matrix(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1],
                                      [1, o ** 1, o ** 2, o ** 3, o ** 4, o ** 5, o ** 6, o ** 7, o ** 8, o ** 9, o ** 10, o ** 11, o ** 12, o ** 13, o ** 14, o ** 15],
                                      [1, o ** 2, o ** 4, o ** 6, o ** 8, o ** 10, o ** 12, o ** 14, o ** 16, o ** 18, o ** 20, o ** 22, o ** 24, o ** 26, o ** 28, o ** 30],
                                        [1, o ** 3, o ** 6, o ** 9, o ** 12, o ** 15, o ** 18, o ** 21, o ** 24, o ** 27, o ** 30, o ** 33, o ** 36, o ** 39, o ** 42, o ** 45],
                                        [1, o ** 4, o ** 8, o ** 12, o ** 16, o ** 20, o ** 24, o ** 28, o ** 32, o ** 36, o ** 40, o ** 44, o ** 48, o ** 52, o ** 56, o ** 60],
                                        [1, o ** 5, o ** 10, o ** 15, o ** 20, o ** 25, o ** 30, o ** 35, o ** 40, o ** 45, o ** 50, o ** 55, o ** 60, o ** 65, o ** 70, o ** 75],
                                        [1, o ** 6, o ** 12, o ** 18, o ** 24, o ** 30, o ** 36, o ** 42, o ** 48, o ** 54, o ** 60, o ** 66, o ** 72, o ** 78, o ** 84, o ** 90],
                                        [1, o ** 7, o ** 14, o ** 21, o ** 28, o ** 35, o ** 42, o ** 49, o ** 56, o ** 63, o ** 70, o ** 77, o ** 84, o ** 91, o ** 98, o ** 105],
                                        [1, o ** 8, o ** 16, o ** 24, o ** 32, o ** 40, o ** 48, o ** 56, o ** 64, o ** 72, o ** 80, o ** 88, o ** 96, o ** 104, o ** 112, o ** 120],
                                        [1, o ** 9, o ** 18, o ** 27, o ** 36, o ** 45, o ** 54, o ** 63, o ** 72, o ** 81, o ** 90, o ** 99, o ** 108, o ** 117, o ** 126, o ** 135],
                                        [1, o ** 10, o ** 20, o ** 30, o ** 40, o ** 50, o ** 60, o ** 70, o ** 80, o ** 90, o ** 100, o ** 110, o ** 120, o ** 130, o ** 140, o ** 150],
                                        [1, o ** 11, o ** 22, o ** 33, o ** 44, o ** 55, o ** 66, o ** 77, o ** 88, o ** 99, o ** 110, o ** 121, o ** 132, o ** 143, o ** 154, o ** 165],
                                        [1, o ** 12, o ** 24, o ** 36, o ** 48, o ** 60, o ** 72, o ** 84, o ** 96, o ** 108, o ** 120, o ** 132, o ** 144, o ** 156, o ** 168, o ** 180],
                                        [1, o ** 13, o ** 26, o ** 39, o ** 52, o ** 65, o ** 78, o ** 91, o ** 104, o ** 117, o ** 130, o ** 143, o ** 156, o ** 169, o ** 182, o ** 195],
                                        [1, o ** 14, o ** 28, o ** 42, o ** 56, o ** 70, o ** 84, o ** 98, o ** 112, o ** 126, o ** 140, o ** 154, o ** 168, o ** 182, o ** 196, o ** 210],
                                        [1, o ** 15, o ** 30, o ** 45, o ** 60, o ** 75, o ** 90, o ** 105, o ** 120, o ** 135, o ** 150, o ** 165, o ** 180, o ** 195, o ** 210, o ** 225]])) / 4

    Lambda = np.linalg.inv(Fourier_16) @ TwoD_Laplace @ Fourier_16
    print(np.round(Lambda.real, 4))
    L = np.round(Lambda.real, 4)
    Twice = np.linalg.inv(Fourier_16) @ Lambda @ Fourier_16
    print(np.round(Twice.real, 4))