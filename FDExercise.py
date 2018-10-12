import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt
import firedrake as fd


def construct3PointStar(N, h):
    fac = 1.0 / (h*h)
    A = fac * (-2.0 * sp.eye(N, k=0) + sp.eye(N, k=1) + sp.eye(N, k=-1))
    A[0, 0] = 1.0
    A[0, 1] = 0.0
    A[N-1, N-2] = 0.0
    A[N-1, N-1] = 1.0
    print(A)
    return A


def exercise1(N = 2**10):
    # Solve the problem u''(x) = 10 sin(20 *x) + cos(x^5)
    # Construct the discrete space:
    X = np.linspace(0, 1, num = N)
    X5 = np.array([x**5 for x in X])
    h = X[1] - X[0]
    A = construct3PointStar(N, h)
    rhs = 10 * np.sin(20 * X) + np.cos(X5)
    rhs[0] = 0
    rhs[N-1] = 0.1
    u = la.spsolve(A, rhs)
    print(u)
    plt.plot(X, u)
    plt.show()


def exercise1Firedrake(N=2**6):
    # Constuct Mesh
    mesh = fd.IntervalMesh(N, 1)
    V = fd.FunctionSpace(mesh, "Lagrange", 1)
    print(V)
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)
    f = fd.Function(V)


def main():
    # exercise1()
    exercise1Firedrake()

if __name__ == "__main__":
    main()
