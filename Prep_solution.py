from sympy import symbols, Eq, solve, nsolve
import numpy as np

def prep_solution_calculator(a, b, c, d, e, f, g):
    sum = a + b + c + d + e + f + g
    sq = np.sqrt(1 / 2)
    c0, c1, c2, c3, c4, c5, s0, s1, s2, s3, s4, s5 = symbols(
        'c0 c1 c2 c3 c4 c5 s0 s1 s2 s3 s4 s5')
    eq1 = Eq(c0 ** 2 + s0 ** 2, 1)
    eq2 = Eq(c1 ** 2 + s1 ** 2, 1)
    eq3 = Eq(c2 ** 2 + s2 ** 2, 1)
    eq4 = Eq(c3 ** 2 + s3 ** 2, 1)
    eq5 = Eq(c4 ** 2 + s4 ** 2, 1)
    eq13 = Eq(c5 ** 2 + s5 ** 2, 1)
    print("here")

    eq6 = Eq(c0 * c1 * c2, np.sqrt(g / sum))
    eq7 = Eq(sq * s0 * (c3 - s3), np.sqrt(c / sum))
    eq8 = Eq(sq * s0 * (s3 + c3), np.sqrt(f / sum))
    eq9 = Eq(sq * c0 * s1 * (c4 - s4), np.sqrt(d / sum))
    eq10 = Eq(sq * c0 * s1 * (s4 + c4), np.sqrt(e / sum))
    eq11 = Eq(sq * c0 * c1 * s2 * (c5 - s5), np.sqrt(a / sum))
    eq12 = Eq(sq * c0 * c1 * s2 * (s5 + c5), np.sqrt(b / sum))
    print("there")

    solution = solve(
        [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13],
        (c0, c1, c2, c3, c4, c5, s0, s1, s2, s3, s4, s5))
    if solution:
        pass
    else:
        initial_guess = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        solution = nsolve(
            [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13],
            (c0, c1, c2, c3, c4, c5, s0, s1, s2, s3, s4, s5), initial_guess)
    return solution


if __name__ == "__main__":
    solutions = prep_solution_calculator(1, 0, 0, 0, 0, 0, 0)
    print(solutions)