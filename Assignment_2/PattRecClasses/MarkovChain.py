import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        cur_state = 0
        S = []
        for i in range(tmax):
            if i == 0:
                dis = DiscreteD(self.q)
                cur_state = dis.rand(nData=1) + 1
                if cur_state == self.nStates + 1:
                    return S
                S.append(cur_state[0])
                
            else:
                row_nested = self.A[cur_state-1]
                row = [item for sublist in row_nested for item in sublist]
                dis = DiscreteD(row)
                cur_state = dis.rand(nData=1) + 1
                if cur_state == self.nStates + 1:
                    return S
                S.append(cur_state[0])
        
        return S
            

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
        '''
        Input:
        matrix pX of size: N (nr of states) x T filled with values proportional to the state-conditional probability mass
        or density values for each state and each frame in the observed feature sequence
        
        Output:
        alpha_hat is matrix of N (nr of states) x T: scaled forward variable
        c is vector of length T and contains the forward scaled factors
        '''
        
        N, T = pX.shape
        alpha_hat = np.zeros((N, T))
        if self.A.shape[0] != self.A.shape[1]:
            # finite duration
            c = np.zeros(T+1)
            A = self.A[:, :-1] # cut exit state
        else:
            # infinite duration
            c = np.zeros(T)
            A = self.A

        # init step:
        alpha_temp = self.q * pX[:,0]
        c[0] = alpha_temp.sum()
        alpha_hat[:,0] = alpha_temp / c[0]
        
        # forward step:
        for t in range(1, T):
            alpha_temp = np.dot(alpha_hat[:,t-1], A) * pX[:,t]
            c[t] = alpha_temp.sum()
            alpha_hat[:,t] = alpha_temp / c[t]

        # termination
        if self.A.shape[0] != self.A.shape[1]:
            # finite duration
            alpha_temp = alpha_hat[:,T - 1] * self.A[:, -1]  # Last column of A for termination
            c[T] = alpha_temp.sum()
        
        return alpha_hat, c

    def finiteDuration(self):
        pass
    
    def backward(self, c, pX):
        '''
        Input:
        matrix pX of size: N (nr of states) x T filled with values proportional to the state-conditional probability mass
        or density values for each state and each frame in the observed feature sequence
        vector c of size: 1 x T as corresponding sequence of scale factors
        
        Output:
        beta_hat is matrix of N (nr of states) x T: scaled backward variable
        
        def backward(self, scaledProbOfObservations, observations, c):
        T = scaledProbOfObservations.shape[1]
        J = self.A.shape[0]
        beta = np.zeros((J, T))
        one = np.ones(J)
        
        #Initialization Step
        if self.is_finite:
            beta[:, T - 1] = self.A[:, J] / (c[T - 1] * c[T])
        else:
            beta[:, T - 1] = one / c[T - 1]


        #Backward Step
        for t in range(T - 2, -1, -1): #Starting with T-1 at index T - 2 
            for i in range(J):
                probThatiCameBeforej = 0
                for j in range(J):
                    probThatiCameBeforej += self.A[i, j] * beta[j, t + 1] * scaledProbOfObservations[j, t + 1]
                beta[i, t] += probThatiCameBeforej
            beta[:, t] = beta[:, t] / c[t]

            
        return beta
        
        
        '''
        N, T = pX.shape
        beta_hat = np.zeros((N, T))
        
        # init step:
        if self.A.shape[0] != self.A.shape[1]:
            # finite duration
            beta_hat[:, T-1] = self.A[:,N]/(c[T-1]*c[T])
        else:
            # infinite duration
            beta_hat[:,T-1] = 1/c[T-1]
        
        # backward step:
        for t in range(T-2, -1, -1): # backwards
            for i in range(N):
                sum = 0
                for j in range(N):
                    sum += self.A[i, j]*beta_hat[j, t+1] * pX[j, t+1]
                beta_hat[i, t] += sum
            beta_hat[:, t] = beta_hat[:, t]/c[t]
        
        return beta_hat

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
