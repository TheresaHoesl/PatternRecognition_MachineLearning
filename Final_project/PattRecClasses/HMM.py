import numpy as np
from .DiscreteD import DiscreteD
from .GaussD import GaussD
from .MarkovChain import MarkovChain


class HMM:
    """
    HMM - class for Hidden Markov Models, representing
    statistical properties of random sequences.
    Each sample in the sequence is a scalar or vector, with fixed DataSize.
    
    Several HMM objects may be collected in a single multidimensional array.
    
    A HMM represents a random sequence(X1,X2,....Xt,...),
    where each element Xt can be a scalar or column vector.
    The statistical dependence along the (time) sequence is described
    entirely by a discrete Markov chain.
    
    A HMM consists of two sub-objects:
    1: a State Sequence Generator of type MarkovChain
    2: an array of output probability distributions, one for each state
    
    All states must have the same class of output distribution,
    such as GaussD, GaussMixD, or DiscreteD, etc.,
    and the set of distributions is represented by an object array of that class,
    although this is NOT required by general HMM theory.
    
    All output distributions must have identical DataSize property values.
    
    Any HMM output sequence X(t) is determined by a hidden state sequence S(t)
    generated by an internal Markov chain.
    
    The array of output probability distributions, with one element for each state,
    determines the conditional probability (density) P[X(t) | S(t)].
    Given S(t), each X(t) is independent of all other X(:).
    
    
    References:
    Leijon, A. (20xx) Pattern Recognition. KTH, Stockholm.
    Rabiner, L. R. (1989) A tutorial on hidden Markov models
    	and selected applications in speech recognition.
    	Proc IEEE 77, 257-286.
    
    """
    def __init__(self, mc, distributions):

        self.stateGen = mc
        self.outputDistr = distributions

        self.nStates = mc.nStates
        self.dataSize = distributions[0].dataSize
    
    def rand(self, nSamples, length=1):
        S = self.stateGen.rand(nSamples)
        X = []
        for i in range(len(S)):
            vec = self.outputDistr[S[i]-1].rand(length)
            X.append(vec[0])
        return X, S
        
        
    def viterbi(self, seq):
        len_seq = seq.shape[0]
        nr_states = self.nStates
        xi = np.zeros((nr_states, len_seq))
        zeta = np.zeros((nr_states, len_seq))
        
        scaled_pX = self.prob(seq, True)
        
        # t = 0
        xi[:,0] = self.stateGen.q * scaled_pX[:,0]
        
        # t = 1, 2, ... T
        for t in range(1, len_seq):
            for j in range(nr_states):
                # Calculate the maximum probability of reaching state j at time t
                max_prob = np.max(xi[:, t-1] * self.stateGen.A[:, j] * scaled_pX[:, t])
                xi[j, t] = max_prob
                # Store the index of the state that achieves that maximum probability
                zeta[j, t] = np.argmax(xi[:, t-1] * self.stateGen.A[:, j] * scaled_pX[:, t])
                    
        # Find state sequence
        states = np.zeros((len_seq))
        
        # Find the state with the maximum probability at time T
        states[-1] = int(np.argmax(xi[:, -1]))
        
        # Backtrack to find the optimal state sequence
        for t in range(len_seq-2, -1, -1):
            states[t] = int(zeta[int(states[t+1]), t+1])
        
        return states

        
    
    def get_q(self, alpha_hat, beta_hat, c):
        gamma = alpha_hat[:,0]*beta_hat[:,0]*c[0]
        q = gamma/sum(gamma)
        return q
    
    def get_A(self, alpha_hat, beta_hat, scaled_pX):
        len_seq = alpha_hat.shape[1]
        nr_states = self.nStates
        xi = np.zeros((nr_states, nr_states, len_seq)) # snd [1]
        for t in range(len_seq-1):
            for currentState in range(nr_states):
                for nextState in range(nr_states):
                    value = alpha_hat[currentState, t] * self.stateGen.A[currentState, nextState] * scaled_pX[nextState, t + 1] * beta_hat[nextState, t + 1]
                    xi[currentState, nextState, t] = value
                        
        xi_bar = np.sum(xi, axis=2)
        xi_sum = np.sum(xi_bar, axis=1)
        A = np.zeros((nr_states, nr_states))
        for i in range(nr_states):
            for j in range(nr_states):
                A[i, j] = xi_bar[i, j]/xi_sum[i]
                
        return A
    
    def get_B(self, alpha_hat, beta_hat, c, seq):
        nr_states = self.nStates
        len_seq = alpha_hat.shape[1]
        gamma = np.zeros((nr_states, len_seq))
        m = np.zeros((nr_states, nr_states))
        co = np.zeros((nr_states, nr_states, nr_states))
        g = np.zeros((nr_states))

        for t in range(len_seq):
            for i in range(nr_states):
                gamma[i, t] = alpha_hat[i, t] * beta_hat[i, t] * c[t]

        for i in range(nr_states):
            for t in range(len_seq):
                m[i] += seq[t,:] * gamma[i, t]
                g[i] += gamma[i, t]
                temp = seq[t,:] - np.atleast_2d(self.outputDistr[i].means)
                co[i] += gamma[i, t] * (temp.T.dot(temp))

        mean = np.zeros((nr_states, nr_states))
        cov = np.zeros((nr_states, nr_states, nr_states))

        for i in range(nr_states):
            if g[i] > 0:
                mean[i] = m[i] / g[i]
                cov[i] = co[i] / g[i]

        return mean, cov

        
    def train(self, seq):
        nr_states = self.nStates
        
        # updating q, A, output distributions
        for i in range(5):
            # get alpha_hats, beta_hats, cs
            scaled_pX = self.prob(seq, True)
            alpha_hat, c = self.stateGen.forward(scaled_pX) # dim: nr_states x T
            beta_hat = self.stateGen.backward(c, scaled_pX) # dim: nr_states x T
            
            # update initial probability vector
            q = self.get_q(alpha_hat, beta_hat, c)
            self.stateGen.q = q
            
            # update transition probability matrix
            A = self.get_A(alpha_hat, beta_hat, scaled_pX)
            self.stateGen.A = A
            
            # update output distributions
            mean, cov = self.get_B(alpha_hat, beta_hat, c, seq)
            self.outputDistr = [GaussD(mean[i], cov=cov[i]) for i in range(len(self.outputDistr))]

    def stateEntropyRate(self):
        pass

    def setStationary(self):
        pass

    def logprob(self, values):
        scaled_pX = self.prob(values, False)
        alpha_hat, c = self.stateGen.forward(scaled_pX)
        return np.sum(np.log(c))

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
    
    def prob(self, values, scaling):
        nr_dim = values.shape[0]
        nr_values = values.shape[0]
        nr_sources = len(self.outputDistr)
        pX = np.zeros((nr_sources, nr_values))
        scaled_pX = np.zeros((nr_sources, nr_values))
        
        for source in range(nr_sources):
            for t in range(nr_values):
                #for i in range(nr_dim):
                    pX[source, t] = self.outputDistr[source].prob(values[t])
                
        if scaling:        
            for source in range(nr_sources):
                for observation in range(nr_values):
                    scaled_pX[source, observation] = pX[source,observation]/np.amax(pX[:,observation])
        else:
            scaled_pX = pX
                
        return scaled_pX