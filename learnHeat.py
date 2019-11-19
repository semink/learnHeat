import numpy as np
import cvxpy as cp
from scipy.sparse import csgraph
from scipy.linalg import expm
from tqdm.notebook import tqdm
from functools import reduce

# Lipshitz constant C1
def lipschitz_c1(L, tau):
    ''' 
    lipschitz_c1(L, tau)
    This function produce the lipschitz constant c1 from Laplacian matrix L and tau.
    '''
    return np.linalg.norm(2*D(L,tau).T@D(L,tau), 'fro')

def descent_condition_by_cost(X, Ltp1, Lt, Htp1, taut, alpha, beta):
    left = cal_cost(X, Ltp1, Htp1, taut, alpha, beta)
    right = cal_cost(X, Lt, Htp1, taut, alpha, beta)
    
    condition = left <= right
    return condition, {"cost_Ltp1":left,"cost_Lt":right}

def descent_condition(X, Ltp1, Lt, Htp1, taut, c2):
    left = Z(X, Ltp1, Htp1, taut)
    right = Z(X, Lt, Htp1, taut) + \
                matrix_inner(Ltp1-Lt,gradient_z_to_L(Lt, X, Htp1, taut)) + \
                c2/2*(np.linalg.norm(Ltp1-Lt, 'fro')**2)
    
    
    condition = left <= right
    return condition, (left,right)

def back_tracking(X, Lt, Htp1, taut, gamma2, alpha, beta, verbose):
    N = X.shape[0]
    S = len(taut)
    eta = 1.1
    c2 = 0.01
    k = 1
    cond = False
    gradient = gradient_z_to_L(Lt, X, Htp1, taut)
    while cond == False:
        c2 = (eta**k)*c2
        dt = gamma2*c2
        Ltp1 = admm(X, Lt, gradient, Htp1, taut, dt, beta, verbose)
        k += 1
        cond, detail = descent_condition(X = X, Ltp1 = Ltp1, Lt = Lt, Htp1 = Htp1, 
                                         taut = taut, c2 = c2)
    return Ltp1

# Lipshitz 
def lipschitz_c3(L, X, H, tau):
# TODO: Consider not to use hessian method..
#     N = L.shape[0]
#     S = len(tau)
#     H_list = H_matrix_to_list(H, N, S)
    
#     cost = []
#     for s, Hs in enumerate(H_list):
#         cost.append(2*np.linalg.norm(Hs, 'fro')*np.linalg.norm(X, 'fro')\
#                     +4*np.linalg.norm(Hs, 'fro')*sum([np.linalg.norm(Hsp, 'fro') for Hsp in H_list]))
#     return np.max(cost)*np.linalg.norm(L,2)**2
    Hessian = hessian_Z_to_tau(X, L, H, tau)
    return np.linalg.norm(Hessian,2)

def soft_threshold(Ht, Lt, X, taut, ct, alpha):
    G = Ht - 1/ct*gradient_z_to_H(Lt, X, Ht, taut)
    Htp1 = np.multiply(np.sign(G), np.maximum(np.abs(G)-alpha/ct, 0))
    return Htp1

def gradient_z_to_H(L, X, H, tau):
    return -2*D(L, tau).T@(X-D(L, tau)@H)

def gradient_z_to_L(L, X, H, tau):
    N = L.shape[0]
    S = len(tau)
    H_list = H_matrix_to_list(H, N, S)
    on_signals = [(-2)*dtrAenLdL(L, A = Hs@X.T, nu = -tau[s]) for s, Hs in enumerate(H_list)]
    off_signal = np.zeros(L.shape)
    for s, Hs in enumerate(H_list):
        off_signals = [dtrAenLdL(L, A = Hsp@Hs.T, nu= -(tau[s]+tau[sp])) for sp, Hsp in enumerate(H_list)]
        off_signal = off_signal + reduce(np.add, off_signals)
    on_signal = reduce(np.add, on_signals)
    g = on_signal + off_signal
    return g

def gradient_z_to_tau(L, X, H, tau):
    N = L.shape[0]
    S = len(tau)
    H_list = H_matrix_to_list(H, N, S)
    g = np.zeros(len(tau))
    for s, Hs in enumerate(H_list):
        on_signal = 2*np.trace(Hs@X.T@L@expm(-tau[s]*L))
        off_signals = [np.trace(Hsp@Hs.T@L@expm(-(tau[s]+tau[sp])*L)) for sp, Hsp in enumerate(H_list)]
        g[s] = on_signal -2*np.sum(off_signals)
    return g

def H_matrix_to_list(H, N, S):
    return np.split(H, [i*N for i in range(1,S)], axis=0)

def dtrAenLdL(L, A, nu):
    if nu == 0:
        return np.zeros(L.shape)
    return nu*dtrAeLdL(L = nu*L, A = A)

def dtrAeLdL(L, A):
    # eigen decomposition
    Eval, Evec = np.linalg.eig(L)
    Eval = np.real(Eval)
    Evec = np.real(Evec)
    # Define B
    B = np.zeros(L.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if i == j:
                B[i,i] = np.exp(Eval[i])
            else:
                B[i,j] = (np.exp(Eval[i])-np.exp(Eval[j])) / (Eval[i]-Eval[j])
    
    # derivative with respect to L
    return Evec@(np.multiply(Evec.T@A.T@Evec,B))@Evec.T

def admm(X, Lt, gradient, Htp1, taut, dt, beta, verbose):
    S = len(taut)
    # variable
    L = cp.Variable(Lt.shape)
    N = Lt.shape[0]
    # constraints
    constraints = [cp.trace(L)== N, L.T==L, L@np.ones((N,1))==np.zeros((N,1))]
    for i in range(N-1):
        constraints += [L[i][i+1:]<=0]
    # objective
    obj = cp.Minimize(cp.trace(gradient.T@(L-Lt)) \
                      + dt/2*(cp.norm(L-Lt, 'fro')**2) + beta*(cp.norm(L, 'fro')**2))
    # solve problem
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose, solver=cp.SCS, scale = 1000, use_indirect = False)
    if L.value is None:
        prob.solve(verbose=verbose, solver=cp.MOSEK)
    return L.value

def Z(X, L, H, tau):
    return np.linalg.norm(X-D(L, tau)@H, 'fro')**2

def tautp1_closed(taut, X, Ltp1, Htp1, et):
    return np.maximum(np.array(taut)-gradient_z_to_tau(Ltp1, X, Htp1, taut)/et, 0)

def D(L, tau):
    return np.concatenate([expm(-tau_s*L) for tau_s in tau], axis=1)

def matrix_inner(A, B):
    return np.trace(B.T@A)

def hessian_Z_to_tau(X, L, H, tau):
    S = len(tau)
    N = L.shape[0]
    
    # Initialize
    Hessian = np.zeros((S,S))
    H_list = H_matrix_to_list(H, N, S)
    for s in range(S):
        Hs = H_list[s]
        for sp in range(S):
            Hsp = H_list[sp]
            if s == sp:
                Hessian[s,s] = -2*np.trace(Hs@X.T@L@L@expm(-tau[s]*L)) + 4*np.trace(Hs@Hs.T@L@L@expm(-tau[s]*L))\
                                + 2* (np.sum([np.trace(H_list[ss]@Hs.T@L@L@expm(-(tau[s]+tau[ss])*L)) for ss in range(S)])\
                                      -np.trace(Hs@Hs.T@L@L@expm(-(tau[s]+tau[s])*L)))
            else:
                Hessian[s,sp] = 2*np.trace(Hsp@Hs.T@L@L@expm(-(tau[s]+tau[sp])*L))
    return Hessian
    
def cal_cost(X, L, H, tau, alpha, beta):
    return Z(X, L, H, tau) + alpha*np.sum(np.abs(H)) + beta*(np.linalg.norm(L,'fro')**2)

def learn_heat(X, 
               L0 = np.array([]), 
               H0 = np.array([]), 
               tau0 = [1,2,3], 
               alpha = 0.1, 
               beta = 0.1, 
               gamma1 = 1.1, 
               gamma2 = 1.1, 
               gamma3 = 1.1, 
               max_iter = 100, 
               verbose=False):
    '''
    learn_heat(X, L0, H0, tau0, alpha, beta, gamma1, gamma2, gamma3, max_iter, verbose)
    
    X: Data matrix. Each column should be an observation
    
    <Optional variables>
    L0: Initial matrix of laplacian matrix
    H0: Initial sparse coefficients
    tau0: Initial taus
    alpha : Sparsity regularization parameter
    beta: Sparsity regularization parameter
    gamma1, gamma2, gamma3: lipshitz constaint factor. Should be larger than 1
    max_iter: Maximum number of iteration. Default is 100.
    verbose: {True, Fale}. In order to see detailed log for all optimization step, please set as True.
    '''
    
    tautp1 = tau0
    N = X.shape[0]
    S = len(tautp1)
    # Initialize laplacian matrix from adjacency matrix
    if L0.shape[0] == 0:
        W = np.random.rand(N,N)
        W = W.T@W
        np.fill_diagonal(W, 0)
        Ltp1 = csgraph.laplacian(W, normed=False)
        Ltp1 = Ltp1/np.trace(Ltp1)*N
    else:
        Ltp1 = L0
        
    if H0.shape[0] == 0:
        Htp1 = np.random.rand(D(Ltp1, tautp1).shape[1], X.shape[1])
    else:
        Htp1 = H0
        
    pbar = tqdm(range(0, max_iter), desc="Learning progress")
    cost_bar = tqdm(total=0, position=1, bar_format='{desc}')
    steps = []
    costs = []
    for t in pbar:
        Lt = Ltp1
        taut = tautp1
        Ht = Htp1    
        # Step for choose c1 (lipschitz)
        ct = gamma1*lipschitz_c1(Lt, taut)
        
        # Update H
        Htp1 = soft_threshold(Ht, Lt, X, taut, ct, alpha)
        cost_bar.set_description_str(f"COST: {'{:0.5f}'.format(cal_cost(X, Lt, Htp1, taut, alpha, beta))} (H updated)")
        # Step to update L and D
        # Update L
        Ltp1 = back_tracking(X, Lt, Htp1, taut, gamma2, alpha, beta, verbose)
        cost_bar.set_description_str(f"COST: {'{:0.5f}'.format(cal_cost(X, Ltp1, Htp1, taut, alpha, beta))} (L updated)")
        
        ## Step to update tau and D
        et = gamma3*lipschitz_c3(Ltp1, X, Htp1, taut)
        tautp1 = tautp1_closed(taut, X, Ltp1, Htp1, et)
        cost_bar.set_description_str(f"COST: {'{:0.5f}'.format(cal_cost(X, Ltp1, Htp1, tautp1, alpha, beta))} (tau updated)")
        steps.append(t)
        costs.append(cal_cost(X, Ltp1, Htp1, tautp1, alpha, beta))
        
    W_learn = -Ltp1
    np.fill_diagonal(W_learn, 0)
    
    learning_curve = {"step": steps, "cost": costs}
    result={"L": Ltp1, "H": Htp1, "tau": tautp1, "W": W_learn, "learning_curve":learning_curve}
    
    return result