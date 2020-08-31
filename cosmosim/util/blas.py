import numpy as np
from scipy.linalg.blas import zhpr, dspr2, zhpmv

def acc_blas(pos, mas, G=1):
    n = mas.size
    # trick: use complex Hermitian to get the packed anti-symmetric
    # outer difference in the imaginary part of the zhpr answer
    # don't want to sum over dimensions yet, therefore must do them one-by-one
    trck = np.zeros((3, n * (n + 1) // 2), complex)
    for a, p in zip(trck, pos.T - 1j):
        zhpr(n, -2, p, a, 1, 0, 0, 1)
        # does  a  ->  a + alpha x x^H
        # parameters: n             --  matrix dimension
        #             alpha         --  real scalar
        #             x             --  complex vector
        #             ap            --  packed Hermitian n x n matrix a
        #                               i.e. an n(n+1)/2 vector
        #             incx          --  x stride
        #             offx          --  x offset
        #             lower         --  is storage of ap lower or upper
        #             overwrite_ap  --  whether to change a inplace
    # as a by-product we get pos pos^T:
    ppT = trck.real.sum(0) + 6
    # now compute matrix of squared distances ...
    # ... using (A-B)^2 = A^2 + B^2 - 2AB
    # ... that and the outer sum X (+) X.T equals X ones^T + ones X^T
    dspr2(n, -0.5, ppT[np.r_[0, 2:n+1].cumsum()], np.ones((n,)), ppT,
          1, 0, 1, 0, 0, 1)
    # does  a  ->  a + alpha x y^T + alpha y x^T    in packed symmetric storage
    # scale anti-symmetric differences by distance^-3
    np.divide(trck.imag, ppT*np.sqrt(ppT), where=ppT.astype(bool),
              out=trck.imag)
    # it remains to scale by mass and sum
    # this can be done by matrix multiplication with the vector of masses ...
    # ... unfortunately because we need anti-symmetry we need to work
    # with Hermitian storage, i.e. complex numbers, even though the actual
    # computation is only real:
    out = np.zeros((3, n), complex)
    for a, o in zip(trck, out):
        zhpmv(n, 0.5, a, mas*-1j, 1, 0, 0, o, 1, 0, 0, 1)
        # multiplies packed Hermitian matrix by vector
    return G * out.real.T