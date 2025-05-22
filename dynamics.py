import numpy as np

S = [i for i in range(16)] #4*4的矩阵
A = [0,1,2,3]
ds_action = {0:-4,1:4,2:-1,3:1}
Pi = 0.25
Gamma = 0.95

def step(s0,a):
    s1 = s0
    if (s0 < 4 and a == 0) or (s0 > 11 and a == 1) or (s0 % 4 == 0 and a == 2) or ((s0 + 1) % 4 == 0 and a == 3) or s0 in [15]:
        pass
    else:
        s1 = s0 + ds_action[a]

    rew = 0 if s0 in [15] else -1

    return s1, rew

def display_v_grid(V):
    for i in range(len(V)):
        print(f'{V[i]:10.3f}', end ='')
        if (i+1) % 4 == 0:
            print('')

def q_value(V,s0,a,gamma = 0.95):
    s1,rew = step(s0,a)
    q = rew + gamma * V[s1]

    return q

def v_value(Pi,V,s0,A,gamma = 0.95):
    vs = 0
    for a in A:
        vs += Pi * q_value(V,s0,a,gamma)

    return vs

def v_eval(Pi,V,S,A,gamma = 0.95,iters = 1):
    for i in range(iters):
        for s in S:
            vs = v_value(Pi,V,s,A,gamma)
            V[s] = vs

    return V

# V = np.zeros(16).tolist()
# v_iter = v_eval(Pi,V,S,A,gamma = Gamma, iters= 120)
# display_v_grid(V)
