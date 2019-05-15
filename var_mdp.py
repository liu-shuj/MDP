import numpy as np
from scipy import sparse
from scipy.sparse.linalg import inv
from numpy.linalg import norm


def genPolicy(P,R,n_iter,pi0,p0,c,gamma):
    # P: distribution of state transition, in format P[state1,action,state2]=Pr(state2|state1,action)
    # R: reward of actions at each state, in format R[state,action]=R(action|state)
    # n_iter: number of iterations of EM
    # c: difference of norm of forward message to converge

    # init
    nState=P.shape[0]
    nAction=P.shape[1]
    alpha=np.zeros((nState*nAction))
    toz=lambda s,a:nAction*s+a
    pi=pi0 # policy
    for iter in range(0,n_iter):
        for s in range(0, nState):
            for a in range(0, nAction):
                alpha[toz(s, a)] = p0[s] * pi[s, a]
        # forward
        lastAlpha=np.ones((nState*nAction))
        pz=sparse.lil_matrix((nState*nAction,nState*nAction))  # state-action trans. dist.
        for s in range(0,nState):
            for a in range(0,nAction):
                pz[toz(s,a)]=pi[s,a]*(P[:,:,s].flatten())
        U=0
        df=1 # discount factor of reward
        count=0
        p_store=[] # p(z_Tau)
        p_store.append(alpha)
        pz=sparse.csr_matrix(pz)
        while (abs(norm(alpha, 2) - norm(lastAlpha, 2)) > c):
            lastAlpha=alpha
            alpha=pz.dot(alpha)
            p_store.append(alpha)
            df=df*gamma
            U=U+(df*R.flatten()*alpha).sum()
            count+=1
        pz=pz.T
        # reversal transition distribution
        pz_rev=(pz.T.multiply(np.outer(alpha,np.reciprocal(alpha,where=alpha!=0)).T)).T

        pz_rev=sparse.csr_matrix(pz_rev)

        # solve final Q(z)
        U=U+(np.power(gamma,count)/(1-gamma)*R.flatten()*alpha).sum()
        lastQ=np.ones((nState*nAction))
        Q=np.zeros((nState*nAction))
        while (abs(norm(Q, 2) - norm(lastQ, 2)) > c):
            lastQ=Q
            Q=alpha*R.flatten()/U+gamma*pz_rev.dot(lastQ)

        # Infinite sum
        Qinf=np.power(gamma,count)/(1-gamma)*Q

        # backward
        new_pi=Q+Qinf
        lastQ=Q
        count=count-1
        while(count>0):
            m=p_store[count+1]!=0
            p_tmp=p_store[count+1]
            p_tmp[m]=1/p_tmp[m]
            cur_rev=(pz.T.multiply(np.outer(p_store[count],p_tmp).T)).T
            cur_rev = sparse.csr_matrix(cur_rev)
            q = p_store[count] * R.flatten() / U
            Q=q+cur_rev.dot(lastQ)
            new_pi=new_pi+Q
            lastQ=Q
            count=count-1
        pi=np.reshape(new_pi,(nState,nAction))
        for i in range(0,pi.shape[0]):
            if(pi[i].sum()!=0):
                pi[i]=pi[i]/pi[i].sum()
        print(U)
        np.save("policy_temp", pi)
    return pi


#P=np.load("P_5_5_40onoff.npy")
#R=np.load("R_5_5_40onoff.npy")
#Re=np.exp(R*1000)
#p0=0.001*np.ones((1000))
#pi=0.25*np.ones((1000,4))
#policy=genPolicy(P,Re,400,pi,p0,0.0001,0.95)
#np.save("policy_fin",policy)
#np.savetxt("policy_fin",policy,fmt='%.5f')
