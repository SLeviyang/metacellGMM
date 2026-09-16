"""
MIXANDMIX (Mixtures by Anderson Mixing) — Python implementation

Implements the core of Cordero-Grande (2020):
  - Mixture-of-populations fixed-point system (Wagner et al. 2012 style):
      e_j(z) = (1/M) tr[ Λ_j ( Σ_k α_k Λ_k / (1 + γ e_k(z))  - z I )^{-1} ]
      m(z)   = (1/M) tr[ ( Σ_k α_k Λ_k / (1 + γ e_k(z))  - z I )^{-1} ]
  - Anderson acceleration (small history, default 2)
  - Homotopy continuation in imaginary part (decrease η from η0 → η_target)

Outputs an estimated density f(x) ≈ (1/π) Im m(x + i η_target).

Assumption: all Λ_k are diagonal covariances and are stored as length-M vectors.

Dependencies: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------
# Utilities
# ---------------------------

def _as_complex(x: np.ndarray) -> np.ndarray:
    return x.astype(np.complex128, copy=False)

def _trace(A: np.ndarray) -> complex:
    return np.trace(A)

def compute_b_plus(m_model, accuracy=0.1, debug=False):

    bp_i = compute_search_interval(m_model)
    if debug:
        print("b_plus_interval: ", bp_i)

    params = SolveParams(warm_start=True)

    x_mm = np.linspace(bp_i[0], bp_i[1], int(bp_i[1]/accuracy))
    _, _, f_mm, _, _ = mixandmix_density(m_model, x_mm,
        params,
    )
    b_plus = np.max(x_mm[f_mm > 1e-4]) + accuracy
    # check that density is 0 at b_plus; if not, keep stepping outward by
    # accuracy until it is, up to a safety cap of 30 iterations
    _, _, f_mm, _, _ = mixandmix_density(m_model, np.array([b_plus]),
        params,
    )
    n_iter = 0
    while f_mm >= 1e-8:
        if n_iter >= 30:
            raise ValueError("Imaginary part of b_plus is not close to 0 after 30 iterations")
        b_plus += accuracy
        _, _, f_mm, _, _ = mixandmix_density(m_model, np.array([b_plus]),
            params,
        )
        n_iter += 1
    if debug:
        print("using b_plus: ", b_plus)
    return b_plus

# find the range of x values for z = x + i*eta over which to compute f(x)
def compute_search_interval(
    m_model,
    t: float = 1.1,
) -> Tuple[float, float]:
    """
    Coarse bounds suggested by the paper for the support.
    We take min eigen over all Λ_k and max eigen over all Λ_k, then apply MP-like spread.
    """
    Lambdas, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]
    K = len(Lambdas)
    if K == 0:
        raise ValueError("Need at least one population covariance matrix.")

    # For diagonal covariances, eigenvalues are the diagonal entries.
    eigmins = []
    eigmaxs = []
    for L in Lambdas:
        eigmins.append(float(np.min(L)))
        eigmaxs.append(float(np.max(L)))

    lam_min = max(min(eigmins), 0.0)  # PSD covariances => nonnegative; clip tiny negatives
    lam_max = max(eigmaxs)

    sg = np.sqrt(gamma)
    a = (1.0 - sg) ** 2 * lam_min / t
    b = (1.0 + sg) ** 2 * lam_max * t

    # If lam_min is 0, a can be 0; keep strictly positive for log grids
    a = max(a, 1e-3)
    b = max(b, a * 1.01)

    return np.array([a, b])


# ---------------------------
# Anderson acceleration
# ---------------------------

@dataclass
class AndersonState:
    m: int = 2              # history size
    lam_factor: float = 0.1 # damping factor multiplier (paper uses 0.1 * max|ΔH|)
    # Stored history:
    e_hist: List[np.ndarray] = None
    g_hist: List[np.ndarray] = None
    h_hist: List[np.ndarray] = None

    def __post_init__(self):
        self.e_hist = []
        self.g_hist = []
        self.h_hist = []


def anderson_step(
    e: np.ndarray,
    g: np.ndarray,
    state: AndersonState,
) -> np.ndarray:
    """
    One Anderson-mixed update based on the last few iterations.
    Follows the spirit of:
      e_{i+1} = g(e_i) - Σ (g_{t}-g_{t-1}) ν
    where ν solves a damped LS problem involving h = g - e and Δh.

    If not enough history, returns plain fixed-point update e_next = g.
    """
    h = g - e

    state.e_hist.append(e.copy())
    state.g_hist.append(g.copy())
    state.h_hist.append(h.copy())

    i = len(state.h_hist) - 1
    if i == 0:
        return g  # first step: no mixing possible

    Qi = min(state.m, i)  # use up to m previous differences
    if Qi <= 0:
        return g

    # Build ΔH = [Δh_{i-Qi+1}, ..., Δh_i]  where Δh_t = h_t - h_{t-1}
    # and ΔG = [Δg_{i-Qi+1}, ..., Δg_i]  where Δg_t = g_t - g_{t-1}
    Dh_cols = []
    Dg_cols = []
    for t in range(i - Qi + 1, i + 1):
        Dh_cols.append(state.h_hist[t] - state.h_hist[t - 1])
        Dg_cols.append(state.g_hist[t] - state.g_hist[t - 1])

    DH = np.column_stack(Dh_cols)  # shape (K, Qi)
    DG = np.column_stack(Dg_cols)  # shape (K, Qi)

    # Damped least squares: ν = argmin || h_i - DH ν ||^2 + λ ||ν||^2
    # Closed form: (DH^H DH + λ I) ν = DH^H h
    # Set λ = 0.1 * max |DH| as in the paper (up to constant); we square-magnitude safe.
    lam = state.lam_factor * float(np.max(np.abs(DH))) if DH.size else 0.0
    Kdim = DH.shape[1]
    A = DH.conj().T @ DH + (lam ** 2) * np.eye(Kdim, dtype=np.complex128)
    b = DH.conj().T @ h

    try:
        nu = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: plain update if ill-conditioned
        return g

    e_next = g - DG @ nu
    return e_next


# ---------------------------
# MIXANDMIX core mapping
# ---------------------------

# Fixed point map given by equation (6) in the paper.
def fixed_point_map(
    e: np.ndarray,
    z: complex,
    m_model,
) -> Tuple[np.ndarray, complex]:
    """
    Given current e (K,), compute:
      A(z,e) = diag( Σ_k α_k Λ_k / (1 + γ e_k) - z )
      g_j    = (1/M) tr( Λ_j A^{-1} )
      m(z)   = (1/M) tr( A^{-1} )

    Returns g (K,), m(z).
    """
    Lambdas, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]
    K = len(Lambdas)
    M = Lambdas[0].shape[0]
    e = _as_complex(e)

    # Build diagonal of A = sum_k (alpha_k / (1 + e_k)) * Lambda_k - gamma*z
    # where e converges to tilde_g from B-G eq (1.5)
    a = np.zeros((M,), dtype=np.complex128)
    for k in range(K):
        denom = (1.0 + e[k])*gamma
        a += (alphas[k] / denom) * _as_complex(Lambdas[k])
    a -= z
    a_inv = 1.0 / a

    g = np.empty(K, dtype=np.complex128)
    g_bar = np.empty(K, dtype=np.complex128)
    for j in range(K):
        g_bar[j] = np.sum(_as_complex(Lambdas[j]) * a_inv) / M
        g[j] = -1.0 / (gamma * z * (1.0 + g_bar[j]))

    # modification to compute B-G/Couilliete's \bar(m) instead of Cordero's m
    m = np.complex128(0.0)
    for i in range(K):
        m += gamma*alphas[i]*g[i]
    #m = np.sum(a_inv) / M
    return g_bar, g, m



def solve_e_and_m(
    x: float,
    e0: np.ndarray,
    m_model,
    p: SolveParams,
    verbose: bool = False,
    outside_spectrum: bool = False,
) -> Tuple[np.ndarray, complex]:
    """
    Solve e(z) at z = x + i*eta_target using homotopy:
      eta: eta0, eta0/beta, ..., eta_target
    Each stage uses Anderson-accelerated fixed-point iterations.
    """
    Lambdas, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]
    e = e0.astype(np.complex128, copy=True)
    eta = float(p.eta0)

    while True:
        if outside_spectrum:
            z = complex(x, 0)
        else:
            z = complex(x, eta)
        state = AndersonState(m=p.anderson_m, lam_factor=p.anderson_lam_factor)
        if verbose:
          print(f"Iteration 0 of {p.max_iter} at eta = {eta}")

        # Iterations at this eta
        for i in range(p.max_iter):
            g_bar, g, m = fixed_point_map(e=e, z=z, m_model=m_model)
            if verbose:
              print(f"Iteration {i} of {p.max_iter} at eta = {eta}")
              print(f"e = {e}")
            #e_next, g, m = fixed_point_map(e=e, z=z, m_model=m_model)
            resid = np.linalg.norm(g_bar - e)
            if outside_spectrum:
                e_next = g_bar
            else:
                e_next = anderson_step(e=e, g=g_bar, state=state)

            if np.linalg.norm(e_next - e) <= p.tol and resid <= p.resid_tol:
                e = e_next
                break

            e = e_next
        if verbose:
          print(f"final e = {e}")

        # If at target eta, finish
        if eta <= p.eta_target * (1.0 + 1e-15):
            # recompute m at final (avoid returning stale m from loop break)
            g_bar, g, m_final = fixed_point_map(e=e, z=complex(x, p.eta_target),
                                         m_model=m_model)
            #g = (g + (1-gamma)/z)/gamma
            return g_bar, g, m_final

        # Decrease eta (homotopy)
        eta = max(eta / p.beta, p.eta_target)


def mixandmix_density(
    m_model,
    x_grid: np.ndarray,
    params: SolveParams,
    e_init: Optional[np.ndarray] = None,
    verbose: bool = False,
    outside_spectrum: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute density on x_grid:
      f(x) ≈ (1/π) Im m(x + i*eta_target)

    Returns: x_grid, f_grid, m_grid
    """
    Lambdas, alphas, gamma = m_model["covariance"], m_model["mixture_frequencies"], m_model["gamma"]
    x_grid = np.asarray(x_grid, dtype=float)
    K = len(Lambdas)
    M = Lambdas[0].shape[0]

    # Basic checks (diagonal covariances stored as vectors)
    for L in Lambdas:
        if L.ndim != 1 or L.shape != (M,):
            raise ValueError("All Λ_k must be 1D vectors with shape (M,).")
    alphas = np.asarray(alphas, dtype=float)
    if alphas.shape != (K,):
        raise ValueError("alphas must have shape (K,).")
    if not np.isclose(np.sum(alphas), 1.0):
        raise ValueError("alphas should sum to 1.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    # Initialize e
    if e_init is None:
        e_init = np.full((K,), 1.0 + 0.0j, dtype=np.complex128)

    e_prev = e_init.copy()
    f = np.zeros((len(x_grid), K), dtype=float)
    mvals = np.empty_like(x_grid, dtype=np.complex128)
    #e_solutions = np.empty((len(x_grid), K), dtype=np.complex128)
    g_solutions = np.empty((len(x_grid), K), dtype=np.complex128)
    g_bar_solutions = np.empty((len(x_grid), K), dtype=np.complex128)

    for i, x in enumerate(x_grid):
        if verbose:
          print(f"Iteration {i} of {len(x_grid)}")
        e0 = e_prev if (params.warm_start and i > 0) else e_init
        g_bar, g, m = solve_e_and_m(
            x=float(x), e0=e0, m_model=m_model, p=params,
            outside_spectrum=outside_spectrum,
        )
        g_solutions[i, :] = g
        g_bar_solutions[i, :] = g_bar
        mvals[i] = m
        for k in range(K):
            f[i,k] = (1.0 / np.pi) * gamma  * np.imag(g[k])
        #f[i,:] = (1.0 / np.pi) * np.imag(m)
        e_prev = g_bar
        if verbose:
          print(f"Density at x = {x} is {f[i]}")

    f_total = f @ alphas

    return x_grid, f, f_total, g_solutions, g_bar_solutions



# def simulate_spectrum(n, p, Lambda_list, alphas, plot=False):
#     K = len(Lambda_list)
#     n_a = [int(n * alphas[i]) for i in range(K)]
#     n_a[-1] = n - sum(n_a[:-1])
#     X_list = []
#     V_list = []
#     for i in range(K):
#         ni = n_a[i]
#         Z = np.random.randn(ni, p)
#         X_list.append(Z * np.sqrt(Lambda_list[i]))
#         V_list.append(np.tile(Lambda_list[i], (ni, 1)))
#     X = np.concatenate(X_list)
#     V = np.concatenate(V_list)
#     if plot:
#         evals = np.linalg.eigvalsh(X.T @ X / p)
#         print(np.max(evals))
#         print(np.min(evals))
#         plt.hist(evals, bins=30)
#         plt.show()
#     return X/np.sqrt(p), V/p


# class that stores the parameters used in the algorithm
@dataclass
class SolveParams:
    # Homotopy in imaginary part: eta0 -> eta_target by dividing by beta
    eta0: float = 1.0
    beta: float = 10.0  # eta is updated by eta = eta / beta
    eta_target: float = 1e-10

    # Fixed-point solve stopping
    max_iter: int = 5000
    tol: float = 1e-10         # convergence in e
    resid_tol: float = 1e-10   # ||g(e)-e|| stopping criterion

    # Anderson
    anderson_m: int = 2
    anderson_lam_factor: float = 0.1

    # Continuation in x: warm-start next x from previous x solution
    warm_start: bool = True
