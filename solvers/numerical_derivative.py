import numpy as np

def approx_fprime(x0, f, eps=1.0e-6, method='2-point'):
    x0 = np.atleast_1d(x0)
    f0 = np.atleast_1d(f(x0))

    shape = f0.shape
    m = len(x0)
    grad = np.empty(shape + (m,))

    h = np.diag(eps * np.ones_like(x0))
    for i in range(m):
        if method == '2-point':
            x = x0 + h[i]
            dx = x[i] - x0[i] # Recompute dx as exactly representable number.
            df = f(x) - f0
        elif method == '3-point':
            x1 = x0 + h[i]
            x2 = x0 - h[i]
            dx = x2[i] - x1[i] # Recompute dx as exactly representable number.
            df = f(x2) - f(x1)
        elif method == 'cs':
            f1 = f(x0 + h[i] * 1.j)
            df = f1.imag
            dx = h[i]
        else:
            raise RuntimeError('method "{method}" is not implemented!')

        grad[..., i] = df / dx

    return np.squeeze(grad)