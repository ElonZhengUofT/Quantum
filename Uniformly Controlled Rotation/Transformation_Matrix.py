import numpy as np

def getM(size):
    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            M[i][j] = (-1) ** (bigj(i, j))

    return M

def bitwise_inner_product(bin1: str, bin2: str) -> int:
    max_length = max(len(bin1), len(bin2))
    bin1 = bin1.zfill(max_length)
    bin2 = bin2.zfill(max_length)

    result = 0
    for i,j in zip(bin1, bin2):
        result += int(i) * int(j)

    return result

def bigj(i: int, j: int) -> int:
    """
    >>> bigj(1,2)
    0
    """
    b = binary(i)
    g = grey_code(j)
    return bitwise_inner_product(b, g)

def grey_code(num: int) -> str:
    """
    >>> grey_code(0)
    '0'
    >>> grey_code(1)
    '1'
    >>> grey_code(2)
    '11'
    >>> grey_code(3)
    '10'
    >>> grey_code(11)
    '1110'
    >>> grey_code(12)
    """
    return bin(num ^ (num >> 1))[2:]

def binary(num: int) -> str:
    """
    >>> binary(0)
    '0'
    >>> binary(1)
    '1'
    >>> binary(2)
    '10'
    >>> binary(3)
    '11'
    """
    return bin(num)[2:]


def get_theta(alpha, M):
    inverse = np.linalg.inv(M)
    return np.dot(inverse, alpha)


if __name__ == '__main__':
    size = 8
    PHI = np.array([(1.9106332362490184 + 3.9252311467094385e-17j),
                    (2.2133936772164815 - 0.22777661338978505j),
                    (2.8549588374993706 - 0.2906413301006183j),
                    (3.464980246312816 - 0.20645507040497738j),
                    (3.727278197046944 - 5.439603600208736e-17j),
                    (3.464980246312816 + 0.2064550704049773j),
                    (2.8549588374993706 + 0.2906413301006183j),
                    (2.2133936772164815 + 0.2277766133897849j)], dtype=complex)
    M = getM(size)
    theta = get_theta(PHI, M)
    print(theta)

