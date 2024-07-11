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
