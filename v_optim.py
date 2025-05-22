import numpy as np
from dynamics import S,A,Gamma
from dynamics import display_v_grid,q_value

def v_optim(V,S,A,gamma = 0.95,iter = 1):
    for i in range(iter):
        for s in S:
            vs = float('-inf')
            for a in A:
                vs = max(vs,q_value(V,s,a,gamma = gamma))
            V[s] = vs

    return V

V = np.zeros(16).tolist()
v_opt = v_optim(V,S,A,gamma = Gamma,iter = 8)
display_v_grid(v_opt)
