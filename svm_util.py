from typing import Tuple, Callable
import numpy as np
import numpy.typing as npt

class ModelTrainingError(Exception):
    '''
    Raised when decision_function or predict is called without the model being trained
    '''
    def __init__(self, message='') -> None:
        super().__init__(message)

def solve_2d_qp(
        alpha_1: np.float64,
        alpha_2: np.float64,
        x1: npt.NDArray,
        x2: npt.NDArray,
        y1: int,
        y2: int,
        Err_1: np.float64,
        Err_2: np.float64,
        C: np.float64,
        kernel_func: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    '''
    Analytically solves the 2D Quadratic subproblem arising in the dual SVM formulation.
    Adapted from Platt (1998).
    '''
    
    if (y1 != y2):
        L = max(0, alpha_2 - alpha_1)
        H = min(C, C + alpha_2 - alpha_1)
    else:
        L = max(0, alpha_2 + alpha_1 - C)
        H = min(C, alpha_2 + alpha_1)
    
    eta = kernel_func(x1, x1) + kernel_func(x2, x2) - 2*kernel_func(x1,x2)
    # print(eta)

    # assuming eta > 0 for now, true unless x1 == x2
    if eta > 0:
        # unconstrained min along the axis
        alpha_2_new =  alpha_2 + (y2 * (Err_1-Err_2) / eta)
    
    # clip unconstrained min of alpha_2 to the edge of box constraint
    if alpha_2_new >= H:
        alpha_2_new = H
    elif alpha_2_new <= L:
        alpha_2_new = L
    else:
        # it must be in the interval (L,H)
        alpha_2_new = alpha_2_new
    
    alpha_1_new = alpha_1 + (y1*y2)*(alpha_2 - alpha_2_new)


    return alpha_1_new, alpha_2_new, eta

def smo_working_set_indices(
        I_up: np.ndarray[int],
        I_low: np.ndarray[int],
        y: np.ndarray[int],
        grads: np.ndarray[np.float64]
    ) -> Tuple[int, int]:
    '''
    Finds a 'maximally violating pair' according to the KKT conditions.
    Returns the index values of the two Lagrange multipliers which comprise
    the working set for the current iteration of SMO.
    '''
    omega_up = -y[I_up] * grads[I_up]
    omega_low = -y[I_low] * grads[I_low] 

    return I_up[omega_up.argmax()], I_low[omega_low.argmin()]

def svm_term_crit(
        I_up: np.ndarray[int],
        I_low: np.ndarray[int],
        grads: np.ndarray[float],
        y: np.ndarray[int],
        tau: float = 0.001
    ) -> bool:
    '''
    Checks KKT conditions (relaxed by tau) for the dual SVM problem.
    Checks m(alpha) - M(alpha) < tau, i.e. there are no more tau-violating pairs of alpha.
    Takahashi (2006).
    '''

    m = np.max(-y[I_up] * grads[I_up])
    M = np.min(-y[I_low] * grads[I_low])

    return (m - M) < tau