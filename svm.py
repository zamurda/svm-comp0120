from typing import List, Callable, Tuple, Optional

import numpy.typing as npt
import numpy as np

from svm.svm_util import (
    ModelTrainingError,
    smo_working_set_indices,
    svm_term_crit,
    solve_2d_qp
)

class SVM:
    '''
    Soft margin SVM with linear or RBF kernel, following an sklearn-like API.
    max_iter = -1 means no limit on the number of iterations.
    '''

    def __init__(
            self,
            C: float = 1.0,
            kernel: str = 'linear',
            gamma = 0.5,  
            random_state = 53,
            max_iter: int = -1,
        ) -> None:

        # initial attributes
        self._train_status = False
        self.C = C
        self.kernel_type = kernel
        self.random_state = random_state
        self.max_iter = max_iter
        self.gamma = gamma

        # svm attributes
        self.support_vectors = None
        self.support_labels = None
        self.support_ = None
        self.dual_variables = None
        self.b = None

        # define linear and RBF kernel functions
        if kernel.lower() == 'linear':
            self.K = lambda x1, x2: x1.T @ x2
        if kernel.lower() == 'rbf':
            self.K = lambda x1, x2: np.exp(-self.gamma * np.linalg.norm(x1-x2, ord=2)**2)
        
        self.training_info = None

        return None

    def fit(self, X: np.ndarray[float], y: np.ndarray[int]) -> 'SVM':
        '''
        Assumes each training sample is a row in X and elements of y take values in {-1,1}.
        '''

        # Reshape y if needed
        y = y.T if len(y.shape) == 2 else y

        # Define index sets (allows for slight relaxation through the use of 'eps'). See Takahashi, 2006.
        # Set eps to 0 for strict KKT.
        index_set = np.arange(0, X.shape[0])
        I_up_boolmask = lambda y, alphas, eps: ((y == 1) & (alphas < self.C - eps)) | ((y == -1) & (alphas > 0 + eps))
        I_low_boolmask  = lambda y, alphas, eps: ((y == -1) & (alphas < self.C - eps)) | ((y == 1) & (alphas > 0 + eps))

        # Initialise alphas, gradients, threshold, and error
        alpha_k = np.zeros((X.shape[0],))
        grads_k = -np.ones((X.shape[0],))
        b_k = -y
        err_k = np.zeros((X.shape[0],))
        s_k = np.zeros((X.shape[0],)) # as defined in Joachims (1998). Used in updating gradient
        self.b = 0

        # Init info dictionary and set train status to true
        self._train_status = True
        self.training_info = {
            'n_iters': 0,
            'etas': [],
            'working_sets': [],
            'changed_alphas': [],
            'alphas': [alpha_k]
        }

        n_iter = 0
        max_iter = np.inf if self.max_iter == -1 else self.max_iter

        while n_iter < max_iter:
            # Construct index set
            eps = 1e-3
            I_up = index_set[I_up_boolmask(y, alpha_k, eps)]
            I_low = index_set[I_low_boolmask(y, alpha_k, eps)]

            # Check stopping condition
            if svm_term_crit(I_up, I_low, grads_k, y):
                self.dual_variables = alpha_k
                self.support_ = index_set[(alpha_k != 0)]
                self.support_vectors = X[index_set[(alpha_k != 0)]]
                self.support_labels = y[index_set[(alpha_k != 0)]]
                break

            else:
                # Formulate and solve 2d subproblem
                i, j = smo_working_set_indices(I_up, I_low, y, grads_k)
                # print(i,j)
                alpha1, alpha2 = alpha_k[i], alpha_k[j]
                alpha_1_new, alpha_2_new, eta = solve_2d_qp(
                    alpha1, alpha2, X[i], X[j], y[i], y[j], err_k[i], err_k[j], self.C, self.K
                    )
                
                # update threshold
                if 0 < alpha_1_new < self.C:
                    b_k = err_k[i] + y[i]*(alpha_1_new-alpha1)*self.K(X[i], X[i]) + y[j]*(alpha_2_new-alpha2)*self.K(X[i], X[j]) + self.b
                elif 0 < alpha_2_new < self.C:
                    b_k = err_k[i] + y[i]*(alpha_1_new-alpha1)*self.K(X[i], X[j]) + y[j]*(alpha_2_new-alpha2)*self.K(X[j], X[j]) + self.b
                else:
                    b1 = err_k[i] + y[i]*(alpha_1_new-alpha1)*self.K(X[i], X[i]) + y[j]*(alpha_2_new-alpha2)*self.K(X[i], X[j]) + self.b
                    b2 = err_k[i] + y[i]*(alpha_1_new-alpha1)*self.K(X[i], X[j]) + y[j]*(alpha_2_new-alpha2)*self.K(X[j], X[j]) + self.b
                    b_k = 0.5*(b1+b2)
                
                # Set updates to self.b, alpha, s, grads, err :)
                self.b = b_k
                alpha_k_new = np.copy(alpha_k)
                alpha_k_new[i], alpha_k_new[j] = alpha_1_new, alpha_2_new
                alpha_k = alpha_k_new
                for t in range(X.shape[0]):
                    # update s_k
                    s_k[t] = s_k[t] + y[i]*(alpha_1_new-alpha1)*self.K(X[t], X[i]) + y[j]*(alpha_2_new-alpha2)*self.K(X[t], X[j])
                grads_k = y*s_k - 1 # check if correct
                err_k = s_k - self.b - y # check if correct

                # Harvest info
                self.training_info['n_iters'] += 1
                self.training_info['etas'].append(eta)
                self.training_info['working_sets'].append([i,j])
                self.training_info['changed_alphas'].append([alpha_1_new, alpha_2_new])
                self.training_info['alphas'].append(alpha_k)

                n_iter += 1
       
        return self

    def predict(self, X: np.ndarray[float]) -> np.ndarray[int]:
        '''
        Predicts labels in {-1, 1} for given samples X
        '''
        if self._train_status:
            if len(X.shape) == 2:
                pred = []
                for x in X:
                    if self.decision_function(x) >= 0:
                        pred.append(1)
                    else:
                        pred.append(-1)
                return np.array(pred)
            else:
                return 1 if self.decision_function(x) >= 1 else -1
        else:
            raise ModelTrainingError('Model needs to be trained before labels can be predicted.')
    
    def decision_function(self, X: np.ndarray[float]) -> np.ndarray[float]:
        '''
        Returns the decision function result for a set of samples X
        '''
        if self._train_status:
            b = self.b
            if len(X.shape) == 2:
                res = []
                for x in X:
                    res.append(np.sum(self.dual_variables[self.support_]*self.support_*self.support_labels*np.array([self.K(x, xj) for xj in self.support_vectors])) - b)
                return np.array(res)
            else:
                return np.sum(self.dual_variables[self.support_]*self.support_*self.support_labels*np.array([self.K(X, xj) for xj in self.support_vectors]) - b)
        else:
            raise ModelTrainingError('Model needs to be trained before decision function can be computed')
    