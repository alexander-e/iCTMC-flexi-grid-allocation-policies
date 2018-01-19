"""
    The opticalfrid Python modulde.

    This Python module can be used to determine blocking probabilities for
    a specific case of a singe link in a flexi-grid optical network, under
    several allocation policies. The used Markov chain models are thoroughly
    introduced in [1]. These models were already partly introduced in [2],
    which on itself was based on [3].

    References
    ----------
    .. [1] Alexander Erreygers, Cristina Rottondi, Giacomo Verticale
           and Jasper De Bock. ``Imprecise Markov Models for Scalable
           and Robust Performance Evaluation of Flexi-Grid Spectrum
           Allocation Policies''. arXiv:?.
    .. [2] Cristina Rottondi, Alexander Erreygers, Giacomo Verticale
           and Jasper De Bock. ``Modelling Spectrum Assignment in a
           Two-Service Flexi-Grid Optical Link with Imprecise
           Continuous-Time Markov Chains''. In: Proceedings of the
           13th International Conference on the Design of Reliable
           Communication Networks (DRCN 2017).
    .. [3] Joobum Kim, Shuyi Yan, Andrea Fumagalli, Eiji Oki and Naoaki
           Yamanaka. ``An analytical model of spectrum fragmentation in
           a two-service elastic optical link''. In: Proceedings of the
           2015 Global Communications Conference (GLOBECOM'15).
    .. [4] Alexander Erreygers and Jasper De Bock. ``Imprecise
           Continuous-Time Markov Chains: Efficient Computational Methods
           with Guaranteed Error Bounds''. In: Proceedings of the Tenth
           International Symposium on Imprecise Probability: Theories and
           Applications.

    Copyright (C) 2018 Alexander Erreygers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

try:
    import gillespy as gpy
except:
    print("The `gillespy.so' file is not found.")
from math import ceil, floor, factorial, isclose
from numbers import Real
import numpy as np
import os
from scipy import sparse
import scipy.sparse.linalg as spla
from functools import partial
from bisect import bisect_left
from time import perf_counter
from datetime import timedelta


def print_header(title, filename):
    """Prints a header to the specified file.

    Parameters
    ----------
    title : str
        The text that should be inside the header,
    filename: str
        The name of the file to which the header should be written.
    """
    f = open(filename, 'a')
    print("\n", file=f)
    print('{:X^80}'.format(''), file=f)
    print('{:x^80}'.format(' ' + title + ' '), file=f)
    print('{:X^80}'.format(''), file=f)
    f.close()


def binomial(n, k):
    """Binomial coefficient (n k): n! / k! / (n-k)!

    Parameters
    ----------
    n: int
    k: int

    Returns
    -------
    int
        The binomial coefficient (n k): n! / k! / (n-k)

    """
    if n == k:
        binom = 1
    elif k == 1:
        binom = n
    else:
        try:
            binom = factorial(n) // factorial(k) // factorial(n-k)
        except ValueError:
            print("Error while determining the binomial coefficient.")
            binom = 0
    return binom


def _cnt(n2, i, K):
    """Evaluate `cnt(i, K)`, as defined by Eqn. (2) in [3].

    Parameters
    ----------
    n2: int
        The number of channels that form a superchannel.
    i: int
        The number of type 1 flows currently allocated.
    K: int
        The number of superchannels either partially or completely
        occupied by type 1 flows: `K = j - e`.

    Returns
    -------
    int
        How many ways `i` balls can be placed in `K` boxes.
    """
    Kmin = ceil(i / n2)
    cnt = sum(
        [(-1)**w * binomial(K, w) * binomial((K-w) * n2, i)
         for w in range(0, K-Kmin+1)])
    return cnt


def _num(n2, i, K, jbar):
    """Evaluate the function `num(i, K, j)`, as defined by Eqn. (4)
    in [3].

    Parameters
    ----------
    n2: int
        The number of channels that form a superchannel.
    i: int
        The number of type 1 flows currently allocated.
    K: int
        The number of superchannels either partially or completely
        occupied by type 1 flows: `K = j - e`.
    jbar: int
        The number of available superchannels, that is `jbar = m2 - j`.

    Returns
    -------
    int
        The number of distinct ways we can distribute `i` type 1 flows
        to make use of exactly `K` out of the `jbar` available superchannels.
    """
    return binomial(jbar, K) * _cnt(n2, i, K)


def _emp(n2, m2, i, j, e):
    """ Evaluate the function `emp(i, K, j)`, as defined by Eqn. (5)
    in [3].

    Parameters
    ----------
    n2: int
        The number of channels that form a superchannel.
    m2: int
        The number of superchannels.
    i: int
        The number of type 1 flows currently allocated.
    j: int
        The number of type 2 flows currently allocated.
    e: int
        The number of completely empty superchannels.

    Returns
    -------
    int
        The total number of cases and possible type 1 departures that would
        leave behind an empty superchannel.
    """
    K = m2 - j - e
    jbar = m2 - j
    return binomial(jbar, 1) * binomial(n2, 1) * _num(n2, i-1, K-1, jbar-1)


def apply_ltro(Qs, numQ, dim, gamble):
    """Apply a lower transition rate operator---that is defined using a
    row of 'extremal' transition rate matrices---to a vector.

    Parameters
    ------ ----
    Qs: array_like
        An array of extremal transition rate matrices.
    numQ: int
        The number of extremal transition rate matrices.
    dim: int
        The dimension of the state space (such that `Qs[i]` is a `dim` x `dim`
        matrix)
    gamble: ndarray
        The vector on which to apply the lower transition rate operator.

    Returns
    -------
    g: ndarray
        The image under the lower transition rate operator.
    """
    g = np.min(np.reshape(Qs.dot(gamble), (numQ, dim)), axis=0)
    return g


def apply_trm(Q, gamble):
    """Apply a transition rate matrix to a vector.

    Parameters
    ------ ----
    Q: ndarray
        The transition rate matrix with dimensions `dim` by `dim`.
    gamble: ndarray
        The `dim`-dimensional vector on which to apply the transition
        rate matrix.

    Returns
    -------
    ndarray
        The image under the transition rate matrix.
    """
    return Q.dot(gamble)


def compute_lower_rate_norm(apply_ltro, dim):
    """Compute the norm of the lower transition rate operator, based on
    Proposition 4 of [4].

    Parameters
    ----------
    apply_ltro: method
        A method that applies that takes as argument a gamble and
        returns the image of the gamble under the lower transition
        rate operator on the state space with `dim` states.
    dim: int
        The number of states in the state space

    Returns
    -------
    float
        The norm of the lower transition rate operator.

    """
    norm = 0
    for i in range(dim):
        ind = np.zeros(dim)
        ind[i] = 1
        norm = max(2 * abs(apply_ltro(ind)[i]), norm)
    return norm


def empirical(apply_ltro, gamble, delta, phi, ul, maxit=10**6):
    """Iteratively determine the limit lower (or upper) expectation of
    a given gamble. This is an implementation of Alg. 1 in [1].

    Parameters
    ----------
    apply_ltro: method
        A method that applies that takes as argument a gamble and
        returns the image of the gamble under the lower transition
        rate operator.
    ganble: ndarray
        The gamble of which to compute the limit lower (or upper)
        expectation.
    delta: float
        The step size used in the iterations.
    phi: float
        The required relative accuracy.
    ul: str
        'L' means lower, 'U' means upper.
    maxit: int, optional
        The maximum number of iterations. The default is `10**6`.

    Returns
    -------
    float
        The estimate of the limit lower (or upper) expectation.
    float
        The achieved relative tolerance (i.e., variation norm divided
        by absolute value of midpoint).
    int
        The number of iterations performed
    timedelta
        The duration of the computations.
    """
    if not ul == 'U':
        g = np.copy(gamble)
    else:
        g = np.copy(-gamble)
    maxG = g.max()
    minG = g.min()
    varNormG = maxG - minG
    midG = (maxG + minG)/2
    numit = 0
    timing = perf_counter()
    while varNormG > phi * abs(midG) and numit < maxit:
        g = np.add(g, delta * apply_ltro(g))
        numit += 1
        maxG = g.max()
        minG = g.min()
        varNormG = maxG - minG
        midG = (maxG + minG)/2
    timing = timedelta(seconds=perf_counter() - timing)
    if ul == 'L':
        ret = midG
    elif ul == 'U':
        ret = - midG
    return ret, varNormG / abs(midG), numit, timing


def steady_state_numerical(
        Q, g_BP1, g_BP2, tol=1e-9, drop_tol=1e-3, fill_factor=10,
        method='ILU-LGMRes'):
    """Determine the limit expectation of a given gamble by solving
    a sparse system of equations.

    Parameters
    ----------
    Q: ndarray
        The transition rate matrix.
    g_BP1: ndarray
        A first gamble of which to compute the limit lower (or upper)
        expectation.
    g_BP2: ndarray
        A second gamble of which to compute the limit lower (or upper)
        expectation.
    tol: float, optional
        Parameter of the `scipy.sparse.linalg` methods. The default is `1e-9`.
    drop_tol: float, optional
        Parameter of the `scipy.sparse.linalg.spilu()` method. The default is
        `1e-3`.
    fill_factor: float, optional
        Parameter of the `scipy.sparse.linalg.spilu()` method. The default is
        `10`.
    method: str, optional
        The method to be used, either 'GMRES', 'LGMRES' or 'BiCGStab',
        corresponding to the respective methods in `scipy.sparse.linalg`.
        If the prefix 'ILU-' is added, the ILU preconditioner is used.
        The default is 'ILU-LGMRES'.

    Returns
    -------
    float
        The estimate of the limit expectation of `g_BP1`.
    float
        The estimate of the limit expectation of `g_BP2`.
    timedelta
        The duration of the set up.
    timedelta
        The duration of the computations excluding the set up time.
    """
    dur_su = perf_counter()
    n = Q.shape[0]
    A = sparse.csc_matrix(Q.transpose(), copy=True)
    A[n-1, :] = np.ones(n)
    b = np.zeros(n)
    b[n-1] = 1

    if method.startswith('ILU-'):
        Mb = spla.spilu(A, drop_tol=drop_tol, fill_factor=fill_factor).solve
    else:
        Mb = lambda x: spla.spsolve(A, x)
    M = spla.LinearOperator((n, n), Mb)

    dur_su = timedelta(seconds=perf_counter() - dur_su)

    dur_c = perf_counter()

    if method.endswith('-GMRes') or ?? method == 'GMRES' ??:
        piEst, info = spla.gmres(A, b, tol=tol, M=M)
    elif method.endswith('BiCGStab'):
        piEst, info = spla.bicgstab(A, b, tol=tol, M=M)
    else:
        piEst, info = spla.lgmres(A, b, tol=tol, M=M)

    dur_c = timedelta(seconds=perf_counter() - dur_c)

    if info is 0:
        bp1 = np.dot(piEst, g_BP1)
        bp2 = np.dot(piEst, g_BP2)
    else:
        bp1 = None
        bp2 = None

    return bp1, bp2, dur_su, dur_c


def gillespie(
        m1, n2, lambda1, lambda2, mu1, mu2, conf_factor, pol="RA",
        batch_size=10**7, min_batches=5, max_batches=50, rel_accur=1e-3,
        seeds=None):
    """Gillespie simulation to determine the blocking probabilities. This simulation
    uses the batch-mean method to increase the accuracy of the estimate.
    This method is actually a Python wrapper for the compiled C code in `gillespy.c`.

    Parameters
    ----------
    m1: int
        The number of channels of the system.
    n2: int
        The number of channels that form a superchannel. `m1` should be a
        multiple of `n2`.
    lambda1: float
        The arrival intensity of type 1 flows.
    lambda2: float
        The arrival intensity of type 2 flows.
    mu1: float
        The intensity of the service process of type 1 flows.
    mu2: float
        The intensity of the service process of type 2 flows.
    conf_factor: float
        The factor used to determine whether or not the estimate is sufficiently
        accurate. Usually derived from the inverse of the standardnormal CDF.
    pol: str
        The allocation policy, either 'RA', 'LF' or 'MF'. The default is
        'RA'.
    batch_size: int
        The number of events per batch. The default is `10**7`.
    min_batches: int
        The minimum number of batches for the batch-mean method, The default
        is `5`.
    max_batches:
        The maximum number of batches for the batch-mean method. The default
        is `50`.
    rel_accur: float
        The required relative accuracy. The default is `1e-3`.
    seeds: array_like
        An array containing 4 floats in the unit interval. The default is `None`.

    Returns
    -------
    float
        The estimated blocking probability of type 1 flows
    float
        The relative accuracy of the aforementioned estimate,
    float
        The estimated blocking probability of type 2 flows
    float
        The relative accuracy of the aforementioned estimate,
    timedelta
        The duration of the simulation.
    """
    try:
        try:
            if len(seeds) is 4:
                useed1 = seeds[0]
                useed2 = seeds[1]
                vseed1 = seeds[2]
                vseed2 = seeds[3]
            else:
                raise Exception('No seed specified.')
        except:
            # We first generate 4 random 64bit integers as seed
            # The best way to do this is to use os.urandom, as this
            # method uses entropy from the system.
            num_bytes = 64 // 8
            useed1 = int.from_bytes(os.urandom(num_bytes), 'big')
            useed2 = int.from_bytes(os.urandom(num_bytes), 'big')
            vseed1 = int.from_bytes(os.urandom(num_bytes), 'big')
            vseed2 = int.from_bytes(os.urandom(num_bytes), 'big')

        # Start the actual computations
        dur = perf_counter()
        bp1, ra1, bp2, ra2 = gpy.gillespie_batchmean(
            m1, n2, lambda1, lambda2, mu1, mu2, pol, batch_size,
            min_batches, max_batches, rel_accur, conf_factor,
            useed1, useed2, vseed1, vseed2)
        dur = timedelta(seconds=perf_counter() - dur)
        if bp1 > 0:
            phi1 = ra1 / bp1
        else:
            phi1 = 0
        if bp2 > 0:
            phi2 = ra2 / bp2
        else:
            phi2 = 0
        ret = bp1, phi1, bp2, phi2, dur
    except:
        ret = _naive_gillespie(
            m1, n2, lambda1, lambda2, m1, mu2,
            num_arrivals=batch_size * min_batches, pol=pol)
    return ret


def _naive_gillespie(
        m1, n2, lambda1, lambda2, mu1, mu2, num_arrivals=10**7, pol='RA'):
    """Naive  pure Python implementation of the Gillespie simulation carried
    out in `gillespie()`, albeit without the batch-mean method.

    Parameters
    ----------
    m1: int
        The number of channels of the system.
    n2: int
        The number of channels that form a superchannel. `m1` should be a
        multiple of `n2`.
    lambda1: float
        The arrival intensity of type 1 flows.
    lambda2: float
        The arrival intensity of type 2 flows.
    mu1: float
        The intensity of the service process of type 1 flows.
    mu2: float
        The intensity of the service process of type 2 flows.
    num_arrivals: int, optional
        The number of  arrivals to simulate. The default is `10**7`.
    pol: str, optional
        The allocation policy, either 'RA', 'LF' or 'MF'. The default is
        'RA'.

    Returns
    -------
    float
        The estimated blocking probability of type 1 flows
    None
        The relative accuracy of the aforementioned estimate,
    float
        The estimated blocking probability of type 2 flows
    None
        The relative accuracy of the aforementioned estimate,
    timedelta
        The duration of the simulation.
    """
    m2 = m1 // n2

    arrivals = 0
    loss1, loss2 = 0, 0

    weights = np.zeros(n2+3)
    weights[0] = lambda1
    weights[1] = lambda2

    state = np.zeros(n2+1)
    state[0] = m2
    numtype1 = np.array(range(0, n2+1))
    timing = perf_counter()

    while arrivals < num_arrivals:
        [u, v] = np.random.rand(2)
        # I = np.sum(state)
        weights[2] = mu2 * (m2 - np.sum(state))
        weights[3:] = mu1 * numtype1[1:] * state[1:]

        cdf = np.cumsum(weights)
        cdf = cdf / cdf[-1]
        the_event = bisect_left(cdf, u) - 2

        if the_event is -2:
            arrivals += 1
            if state[0] is 0:
                loss2 += 1
            if np.sum(state[0:-1]) < 1:
                loss1 += 1
            else:
                if pol == 'RA':
                    _cdf = np.cumsum((n2 - numtype1[:-1]) * state[:-1])
                    _cdf = _cdf / _cdf[-1]
                    alloc = bisect_left(_cdf, v)
                else:
                    _nonzero = np.nonzero(state[1:-1])
                    try:
                        if pol == 'LF':
                                alloc = _nonzero[0][0]+1
                        elif pol == 'MF':
                            alloc = _nonzero[0][-1]+1
                        else:
                            print("Error while selecting policy.")
                    except:
                        alloc = 0

                state[alloc] -= 1
                state[alloc+1] += 1
        elif the_event is -1:
            arrivals += 1
            if np.sum(state[0:-1]) < 1:
                loss1 += 1
            if state[0] < 1:
                loss2 += 1
            else:
                state[0] -= 1
        elif the_event is 0:
            state[0] += 1
        else:
            state[the_event] -= 1
            state[the_event-1] += 1
    timing = timedelta(s=perf_counter() - timing)
    try:
        bp1 = loss1 / arrivals
        bp2 = loss2 / arrivals
    except:
        bp1 = None
        bp2 = None
    return bp1, None, bp2, None, timing


class StateSpace:
    """A state space.

    The `StateSpace` class contains methods to generate a state space, be it
    the detailed or the reduced, of the single optical link under study in [1].
    """

    def __init__(self, m1, n2, reduced=False):
        """Initialise a StateSpace.

        Parameters
        ----------
        m1: int
            The number of channels.
        n2: int
            The number of channels that form a superchannel. `m1` should
            be a multiple of `n2`.
        reduced: boolean, optional
            Whether or not to use the reduced state space. The default is
            `False`.
        """
        self.m1 = m1
        self.n2 = n2
        self.m2 = m1 // n2
        if reduced:
            self.red = True
            self.initialise_reduced_state_space(m1, n2)
        else:
            self.red = False
            self.fact_dict = None
            self.initialise_detailed_state_space(m1, n2)

    def initialise_reduced_state_space(self, m1, n2):
        """In In initialise a reduced state space.

        Parameters
        ----------
        m1: int
            The number of channels.
        n2: int
            The number of channels that form a superchannel. `m1` should
            be a multiple of `n2`.
        """
        m2 = self.m2
        spaceTemp = []
        for i in range(0, m1+1):
            for j in range(0, min((m1 - i)//n2+1, m2+1)):
                for e in range(max(0, m2-i-j), min((m1 - i)//n2-j+1, m2+1)):
                    if i + j + e >= m2 and i + (j + e) * n2 <= m1:
                        spaceTemp.append((i, j, e))

        # Generate the fact_dict
        fact_dict = []
        for (i, j, e) in spaceTemp:
            if (i-1, j, e) in spaceTemp \
                    or (i-1, j, e+1) in spaceTemp:
                fact = _emp(n2, m2, i, j, e) / _num(n2, i, m2-j-e, m2-j)
                fact_dict.append(((i, j, e), fact))
        self.fact_dict = dict(fact_dict)

        # Generate the actual state space
        self.space = []
        space_dict = []
        for ell, tup in enumerate(spaceTemp):
            x = tup
            self.space.append(x)
            space_dict.append((x, ell))

        self.dim = len(self.space)  # equals k-1 + 1 = k
        self.space_dict = dict(space_dict)

        # Generating relevant gambles.
        self.g_BP = np.zeros(self.dim)
        # BP1
        self.g_BP1 = np.zeros(self.dim)
        for k, (i, j, e) in enumerate(self.space):
            if (m1 > 0 and m1 == i + j * n2):
                self.g_BP[k] = 1
                self.g_BP1[k] = 1
        # BP2
        self.g_BP2 = np.zeros(self.dim)
        for k, (i, j, e) in enumerate(self.space):
            if (m1 > 0 and e == 0):
                self.g_BP[k] = 1
                self.g_BP2[k] = 1

    def initialise_detailed_state_space(self, m1, n2):
        """Initialise a detailed state space.

        Parameters
        ----------
        m1: int
            The number of channels.
        n2: int
            The number of channels that form a superchannel. `m1` should
            be a multiple of `n2`.
        """
        m2 = self.m2
        space = []
        for _k in range(n2+1):
            k = n2 - _k
            if k is n2:
                for in2 in range(0, m2+1):
                    space.append((in2,))
            elif k > 0:
                space_old = space[:]
                space = []
                for tup in space_old:
                    upbound = m2-sum(tup)
                    for ik in range(upbound+1):
                        space.append((ik,) + tup)
            else:
                space_old = space
                space = []
                for tup in space_old:
                    upbound = m2-sum(tup)
                    for i0 in range(upbound+1):
                        newstate = (i0,) + tup
                        space.append(newstate)
        self.space = []
        space_dict = []
        for ell, tup in enumerate(space):
            x = tup
            self.space.append(x)
            space_dict.append((x, ell))
        self.space_dict = dict(space_dict)
        self.dim = len(self.space)

        # Generating relevant gambles.
        # BP1
        self.g_BP1 = np.zeros(self.dim)
        for k, x_e in enumerate(self.space):
            R = sum([x_e[_l] * (n2 - _l) for _l in range(n2)])
            if (m1 > 0 and R == 0):
                self.g_BP1[k] = 1
        # BP2
        self.g_BP2 = np.zeros(self.dim)
        for k, x_e in enumerate(self.space):
            if (m1 > 0 and x_e[0] == 0):
                self.g_BP2[k] = 1

    def construct_ltro(self, mu1, mu2, lambda1, lambda2, pol=None, impr=False):
        """Construct a lower transition rate operator,

        Parameters
        ----------
        mu1: float
            The intensity of the service process of type 1 flows.
        mu2: float
            The intensity of the service process of type 2 flows.
        lambda1: float
            The arrival intensity of type 1 flows.
        lambda2: float
            The arrival intensity of type 2 flows.
        pol: str, optional
            The allocation policy, either 'RA', 'LF' or 'MF' for the detailed
            state space or 'R', 'LM' or `None` for the reduced state space.
            The default is `None`.
        impr: boolean, optional
            Whether to use an imprecise or precise lower transition rate
            operator. The default is `False`.

        Returns
        -------
        ltro: method
            The lower transition rate operator
        normQ: float
            The norm of the lower transition rate operator.
        """
        if pol not in ['RA', 'LF', 'MF', 'R', 'LM', None]:
            pass
        if impr:
            Qs, numQ, normQ = self.construct_Q_matrices(
                mu1, mu2, lambda1, lambda2, pol)
            ltro = partial(apply_ltro, Qs, numQ, self.dim)
        else:
            # Construct the transition rate matrix
            Q, normQ = self.construct_Q_matrix(mu1, mu2, lambda1, lambda2, pol)
            ltro = partial(apply_trm, Q)
        return ltro, normQ

    def construct_Q_matrix(self, mu1, mu2, lambda1, lambda2, pol):
        """Construct a transition rate matrix.

        Parameters
        ----------
        mu1: float
            The intensity of the service process of type 1 flows.
        mu2: float
            The intensity of the service process of type 2 flows.
        lambda1: float
            The arrival intensity of type 1 flows.
        lambda2: float
            The arrival intensity of type 2 flows.
        pol: str
            The allocation policy, either 'RA', 'LF', 'MF', 'R', 'LM'
            or `None`.

        Returns
        -------
        Q: ndarray
            The transition rate matrix.
        normQ: float
            The norm of the transition rate matrix.
        """
        if self.red:
            Qvals = []
            col_ind = []
            row_ind = []
            m1 = self.m1
            m2 = self.m2
            n2 = self.n2
            normQ = 0
            for x, (i, j, e) in enumerate(self.space):
                R = m1 - i - j*n2
                sumrates = 0
                # Departure of type 2
                if j > 0:
                    y = self.space_dict[(i, j-1, e+1)]
                    Qvals.append(j * mu2)
                    col_ind.append(y)
                    row_ind.append(x)
                    sumrates += j * mu2
                # Arrival of type 2
                if e > 0:
                    y = self.space_dict[(i, j+1, e-1)]
                    Qvals.append(lambda2)
                    col_ind.append(y)
                    row_ind.append(x)
                    sumrates += lambda2
                # Arrival of type 1
                if R > 0:
                    sumrates += lambda1
                    if pol == 'R':
                        lambdaP = lambda1 * e * n2 / R
                        if (i+1, j, e-1) in self.space_dict:
                            y = self.space_dict[(i+1, j, e-1)]
                            Qvals.append(lambdaP)
                            col_ind.append(y)
                            row_ind.append(x)
                        if (i+1, j, e) in self.space_dict:
                            y = self.space_dict[(i+1, j, e)]
                            Qvals.append(lambda1-lambdaP)
                            col_ind.append(y)
                            row_ind.append(x)
                    elif pol == 'LM':
                        if i == (m2 - j - e) * n2:
                            y = self.space_dict[(i+1, j, e-1)]
                            Qvals.append(lambda1)
                            col_ind.append(y)
                            row_ind.append(x)
                        else:
                            y = self.space_dict[(i+1, j, e)]
                            Qvals.append(lambda1)
                            col_ind.append(y)
                            row_ind.append(x)
                    else:
                        print("Error with selecting the policy")
                        break
                # Departures of type 1
                if (i-1, j, e+1) in self.space_dict \
                        or (i-1, j, e) in self.space_dict:
                    sumrates += i*mu1
                    fact = self.fact_dict[(i, j, e)]
                    if (i-1, j, e+1) in self.space_dict:
                        y = self.space_dict[(i-1, j, e+1)]
                        Qvals.append(fact*mu1)
                        col_ind.append(y)
                        row_ind.append(x)
                    if (i-1, j, e) in self.space_dict:
                        y = self.space_dict[(i-1, j, e)]
                        Qvals.append((i-fact)*mu1)
                        col_ind.append(y)
                        row_ind.append(x)
                Qvals.append(-sumrates)
                normQ = max(normQ, 2 * sumrates)
                col_ind.append(x)
                row_ind.append(x)
            Q = sparse.csr_matrix(
                (Qvals, (row_ind, col_ind)), shape=(self.dim, self.dim))
        else:
            Qvals = []
            col_ind = []
            row_ind = []
            n2 = self.n2
            m2 = self.m2
            normQ = 0
            for x, x_e in enumerate(self.space):
                I = sum(x_e)  # i0 + i1 + ... + i_{n_2}
                R = sum([x_e[k] * (n2 - k) for k in range(n2)])
                sumrates = 0
                # Departure of type 2
                y_e = (x_e[0]+1,) + x_e[1:]
                if y_e in self.space_dict:
                    y = self.space_dict[y_e]
                    Qvals.append((m2 - I) * mu2)
                    row_ind.append(x)
                    col_ind.append(y)
                    sumrates += (m2 - I) * mu2
                # Arrival of type 2
                # Sufficient condition for the existence of such a state
                if x_e[0] > 0:
                    y_e = (x_e[0]-1,) + x_e[1:]
                    y = self.space_dict[y_e]
                    Qvals.append(lambda2)
                    row_ind.append(x)
                    col_ind.append(y)
                    sumrates += lambda2
                # Arrival of type 1
                if R > 0:
                    sumrates += lambda1
                    if pol == 'RA':
                        for k in range(0, n2):
                            # We implicitly assume that n2 >= 2
                            y_e = x_e[0:k] + (x_e[k]-1, x_e[k+1]+1) + x_e[k+2:]
                            if y_e in self.space_dict:
                                y = self.space_dict[y_e]
                                Qvals.append(lambda1 * x_e[k] * (n2 - k) / R)
                                row_ind.append(x)
                                col_ind.append(y)
                    elif pol == 'LF':
                        iLF = 0
                        for i in range(1, n2):
                            if x_e[i] > 0:
                                iLF = i
                                break
                        y_e = x_e[0:iLF] + (x_e[iLF]-1, x_e[iLF+1]+1) + x_e[iLF+2:]
                        if y_e in self.space_dict:
                            y = self.space_dict[y_e]
                            Qvals.append(lambda1)
                            row_ind.append(x)
                            col_ind.append(y)
                    elif pol == 'MF':
                        iMF = 0
                        for i in range(1, n2):
                            if x_e[n2-i] > 0:
                                iMF = n2-i
                                break
                        y_e = x_e[0:iMF] + (x_e[iMF]-1, x_e[iMF+1]+1) + x_e[iMF+2:]
                        if y_e in self.space_dict:
                            y = self.space_dict[y_e]
                            Qvals.append(lambda1)
                            row_ind.append(x)
                            col_ind.append(y)
                    else:
                        print("Error with selecting the policy.")
                        break
                # Departures of type 1
                for k in range(1, n2+1):
                    # Sufficient condition for the existence of such a state
                    if x_e[k] > 0:
                        y_e = x_e[0:k-1] + (x_e[k-1]+1, x_e[k]-1) + x_e[k+1:]
                        y = self.space_dict[y_e]
                        Qvals.append(k * x_e[k] * mu1)
                        row_ind.append(x)
                        col_ind.append(y)
                        sumrates += k * x_e[k] * mu1
                Qvals.append(-sumrates)
                normQ = max(normQ, 2 * sumrates)
                row_ind.append(x)
                col_ind.append(x)
            Q = sparse.csr_matrix(
                (Qvals, (row_ind, col_ind)), shape=(self.dim, self.dim))
        return Q, normQ

    def construct_Q_matrices(self, mu1, mu2, lambda1, lambda2, pol):
        """Construct a row of extremal transition rate matrices.

        Parameters
        ----------
        mu1: float
            The intensity of the service process of type 1 flows.
        mu2: float
            The intensity of the service process of type 2 flows.
        lambda1: float
            The arrival intensity of type 1 flows.
        lambda2: float
            The arrival intensity of type 2 flows.
        pol: str
            The allocation policy, either 'R', 'LM' or None.

        Returns
        -------
        Qs: array_like
            An array containing the extremal transition rate matrices.
        numQ: int
            The number of extremal transition rate matrices, i.e. the
            number of elements of `Qs`.
        normQ: float
            The norm of the lower transition rate operator.
        """
        if not self.red:
            return None
        if pol in ['R', 'LM']:
            numQ = 2
        else:
            numQ = 4
        m1 = self.m1
        m2 = self.m2
        n2 = self.n2

        Qvals = []
        col_ind = []
        row_ind = []
        Q_bis = [[] for _ in range(numQ)]
        row_bis = [[] for _ in range(numQ)]
        col_bis = [[] for _ in range(numQ)]
        normQ = 0
        for x, (i, j, e) in enumerate(self.space):
            R = m1 - i - j*n2
            sumrates = 0
            # Departure of type 2
            if j > 0:
                y = self.space_dict[(i, j-1, e+1)]
                Qvals.append(j * mu2)
                col_ind.append(y)
                row_ind.append(x)
                sumrates += j*mu2
            # Arrival of type 2
            if e > 0:
                y = self.space_dict[(i, j+1, e-1)]
                Qvals.append(lambda2)
                col_ind.append(y)
                row_ind.append(x)
                sumrates += lambda2
            # Arrival of type 1
            if R > 0:
                sumrates += lambda1
                ya = (i+1, j, e)
                yb = (i+1, j, e-1)
                feasa = ya in self.space_dict
                feasb = yb in self.space_dict
                if feasa and not feasb:
                    Qvals.append(lambda1)
                    col_ind.append(self.space_dict[ya])
                    row_ind.append(x)
                elif feasb and not feasa:
                    Qvals.append(lambda1)
                    col_ind.append(self.space_dict[yb])
                    row_ind.append(x)
                else:
                    if numQ == 4:
                        for _k in range(numQ):
                            if _k < 2:
                                Q_bis[_k].append(lambda1)
                                col_bis[_k].append(self.space_dict[ya])
                                row_bis[_k].append(x)
                            else:
                                Q_bis[_k].append(lambda1)
                                col_bis[_k].append(self.space_dict[yb])
                                row_bis[_k].append(x)
                    elif pol == 'R':
                        lambdaP = lambda1 * e * n2 / R
                        Qvals.append(lambdaP)
                        col_ind.append(self.space_dict[yb])
                        row_ind.append(x)
                        Qvals.append(lambda1-lambdaP)
                        col_ind.append(self.space_dict[ya])
                        row_ind.append(x)
                    elif pol == 'LM':
                        if i == (m2 - j - e) * n2:
                            Qvals.append(lambda1)
                            col_ind.append(self.space_dict[yb])
                            row_ind.append(x)
                        else:
                            Qvals.append(lambda1)
                            col_ind.append(self.space_dict[ya])
                            row_ind.append(x)
                    else:
                        print("ERROR WHILE SELECTING THE POLICY")
                        break
            # Departures of type 1
            if (i-1, j, e+1) in self.space_dict \
                    or (i-1, j, e) in self.space_dict:
                sumrates += i*mu1
                ya = (i-1, j, e+1)
                yb = (i-1, j, e)
                feasa = ya in self.space_dict
                feasb = yb in self.space_dict
                if feasa and not feasb:
                    Qvals.append(i*mu1)
                    col_ind.append(self.space_dict[ya])
                    row_ind.append(x)
                elif feasb and not feasa:
                    Qvals.append(i*mu1)
                    col_ind.append(self.space_dict[yb])
                    row_ind.append(x)
                else:
                    imin = max(0, 2*(m2 - j - e) - i)
                    imax = min(m2 - j - e,
                               floor((n2*(m2 - j - e) - i)/(n2 - 1)))
                    for _k in range(numQ):
                        if _k == 0 or _k == 2:
                            Q_bis[_k].append(imin*mu1)
                            col_bis[_k].append(self.space_dict[ya])
                            row_bis[_k].append(x)
                            Q_bis[_k].append((i-imin)*mu1)
                            col_bis[_k].append(self.space_dict[yb])
                            row_bis[_k].append(x)
                        else:
                            Q_bis[_k].append(imax*mu1)
                            col_bis[_k].append(self.space_dict[ya])
                            row_bis[_k].append(x)
                            Q_bis[_k].append((i-imax)*mu1)
                            col_bis[_k].append(self.space_dict[yb])
                            row_bis[_k].append(x)
            Qvals.append(-sumrates)
            col_ind.append(x)
            row_ind.append(x)
            normQ = max(normQ, 2 * sumrates)
        _Qvals = []
        _rows = []
        _cols = []
        for i in range(numQ):
            _Qvals.extend(Qvals[:])  # Creates a copy
            _Qvals.extend(Q_bis[i][:])
            _rows.extend([r + i * self.dim for r in row_ind])
            _rows.extend([r + i * self.dim for r in row_bis[i]])
            _cols.extend(col_ind[:])
            _cols.extend(col_bis[i][:])
        Qs = sparse.csr_matrix(
            (_Qvals, (_rows, _cols)), shape=(numQ * self.dim, self.dim))
        return Qs, numQ, normQ
