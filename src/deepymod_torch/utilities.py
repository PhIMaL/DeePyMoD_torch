from itertools import product, combinations


def string_matmul(list_1, list_2):
    prod = [element[0] + element[1] for element in product(list_1, list_2)]
    return prod


def terms_definition(poly_list, deriv_list):
    theta_uv = [string_matmul(u, v) for u, v in combinations(poly_list, 2)]
    theta_dudv = [string_matmul(du, dv)[1:] for du, dv in combinations(deriv_list, 2)]
    theta_udv = [string_matmul(u[1:], dv[1:]) for u, dv in product(poly_list, deriv_list)]
    theta = [element for theta_specific in (theta_uv + theta_dudv + theta_udv) for element in theta_specific]

    return theta
