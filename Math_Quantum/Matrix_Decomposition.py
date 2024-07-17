import sympy as sp
import numpy as np
import time
import matplotlib.pyplot  as plt

def SVD(A: sp.Matrix):
    """
    Singular Value Decomposition
    :param A:
    :return:
    """
    time_start = time.time()
    AtA = A.T * A
    eigenvects, eigenvals = AtA.diagonalize()
    singular_values = sp.sqrt(eigenvals)
    V = eigenvects
    S = singular_values
    S_inv = S.inv()
    time_end = time.time()
    print('SVD cost', time_end - time_start, 's')
    AV = A * V
    time_end = time.time()
    print('SVD cost', time_end - time_start, 's')
    U = AV * S_inv
    return U, S, V


if __name__ == '__main__':
    file = open('matrix.txt', 'a+')
    a, b, c, d, k = sp.symbols('a b c d k')
    A = sp.Matrix([[a,b,0,b],
                         [b,a,k * b,0],
                         [0,k * b,a, b],
                         [b,0,b,a]])
    file.seek(0)
    read = file.read()
    if read.find(str(A)) != -1:
        U = sp.Matrix(sp.sympify(read[read.find("U = ") + 4: read.find("S = ") - 1]))
        S = sp.Matrix(sp.sympify(read[read.find("S = ") + 4: read.find("V = ") - 1]))
        V = sp.Matrix(sp.sympify(read[read.find("V = ") + 4:]))
        file.close()
    else:
        U, S, V = SVD(A)
        print(U)
        print(S)
        print(V)
        # 把矩阵写入文件
        file.write("For matrix = " + str(A) + "svd decomposition is:\n")
        file.write("U = " + str(U) + "\n")
        file.write("S = " + str(S) + "\n")
        file.write("V = " + str(V) + "\n")
        file.close()
    dict = {a: 1, b: 2, c: 3, d: 4, k: 3}
    A_n = A.subs(dict)
    plt.imshow(np.array(A_n).astype(np.float64))
    plt.colorbar()
    plt.show()
    U_n, S_n, V_n = np.linalg.svd(np.array(A_n).astype(np.float64))
    print(U_n)
    plt.imshow(U_n)
    plt.colorbar()
    plt.show()




