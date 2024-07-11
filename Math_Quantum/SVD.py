import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import sympy as sp

def is_unitary(U):
    """
    Check if a matrix is unitary
    :param U:
    :return:
    """
    return np.allclose(np.eye(U.shape[0]), U @ U.T.conj()) and np.allclose(np.eye(U.shape[0]), U.T.conj() @ U)

def is_diagonal(D):
    """
    Check if a matrix is diagonal
    :param D:
    :return:
    """
    return np.allclose(np.diag(np.diag(D)), D)

def SVD(A):
    """
    Singular Value Decomposition
    :param A:
    :return:
    """
    U, S, V = nla.svd(A)
    return U, S, V


def PolarD(A):
    """
    Polar Decomposition
    :param A:
    :return:
    """
    U, R = sla.polar(A)
    return U, R


def Chelosky(A):
    """
    Chelosky Decomposition
    :param A:
    :return:
    """
    L = np.linalg.cholesky(A)
    return L

def spectral_decomposition(A):
    """
    Spectral Decomposition,return V,D, where V is the matrix of eigenvectors and D is the diagonal matrix of eigenvalues
    :param A:
    :return:
    """
    eigvals, eigvecs = np.linalg.eig(A)
    V = eigvecs
    D = np.diag(eigvals)
    return V, D


if __name__ == '__main__':
    order = input("Enter the order of test: ")
    a, b, c, d = sp.symbols('a b c d')
    Matrixs = [sp.Matrix([[a,b,0,b],
                         [b,a,b+c,0],
                         [0,b+c,a,b],
                         [b,0,b,a]])]

    for A in Matrixs:
        # test svd
        if order == "svd":
            # test SVD
            U, S, V = A.singular_value_decomposition()
            print(U)
            print(S)
            print(V)

        # test polar decomposition
        if order == "polar":
            # test Polar Decomposition
            U, R = PolarD(A)
            print(U)
            print(R)
            if is_unitary(U):
                print("U is unitary")
            else:
                print("U is not unitary")

        # test polar decomposition
        if order == "spectral":
            # test Polar Decomposition
            V, D = spectral_decomposition(A)
            print(V)
            print(D)
            if is_unitary(V):
                if is_diagonal(D):
                    print("spectral decomposition is correct")
                else:
                    print("D is not diagonal")
            else:
                print("V is not unitary")


