from numpy import array, ndarray, ones, zeros
from scipy.special import factorial
from scipy.linalg import solve


def derivative_operator(radius: ndarray, order: int=1) -> ndarray:
    A = zeros([radius.size, radius.size])
    for i in range(1, radius.size - 1):
        A[i, i - 1 : i + 2] = findiff_coeffs(
            radius[i - 1 : i + 2] - radius[i], order=order
        )
    n = 2 + order
    A[0, :n] = findiff_coeffs(radius[:n] - radius[0], order=order)
    A[-1, -n:] = findiff_coeffs(radius[-n:] - radius[-1], order=order)
    return A


def findiff_coeffs(points: ndarray, order: int=1) -> ndarray:
    # check validity of inputs
    if type(points) is not ndarray:
        points = array(points)
    n = len(points)
    if n <= order:
        raise ValueError(
            "The order of the derivative must be less than the number of points"
        )
    # build the linear system
    b = zeros(n)
    b[order] = factorial(order)
    A = ones([n, n])
    for i in range(1, n):
        A[i, :] = points**i
    # return the solution
    return solve(A, b)