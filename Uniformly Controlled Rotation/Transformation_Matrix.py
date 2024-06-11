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
    PHI = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])
    M = getM(size)
    M2 = np.dot(M, M)
    print(M2)
    print(M)
    theta = get_theta(PHI, M)
    print(theta)

    size = 4
    PHI = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
    M = getM(size)
    M2 = np.dot(M, M)
    print(M2)
    print(M)
    theta = get_theta(PHI, M)
    print(np.linalg.inv(M))
    print(theta)

