from DiscreteD import DiscreteD
from GaussD import GaussD
from MarkovChain import MarkovChain
import numpy as np

x = np.array( [ 5, 1, 9 ,4] )
g1 = GaussD( means=[0], stdevs=[1])
print(x)

disc = DiscreteD(x)
print(disc.rand(1))

mc = MarkovChain( np.array( [ 1, 0, 0 ] ), np.array( [ [ 0.75, 0.2, 0.05 ], [ 0.25, 0.7, 0.05 ] ] ) )
print(mc.rand(100))