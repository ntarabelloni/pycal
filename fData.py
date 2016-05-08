import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy import linalg, sparse

# A simple class for functional data objects
class fData(object) :
    def __init__( self, grid = None, data = None ) :
        self.grid = grid.copy()
        self.data = data.copy()

    def plot( self ) :
        plt.figure()
        [ plt.plot( self.grid, self.data[ i, : ]) for i in range( 0, N ) ]
        plt.show()

# A simple class that implements an Exponential Covariance function
class ExpCov(object) :

    def __init__( self, alpha, beta ) :
        self.alpha = alpha
        self.beta = beta

    def eval( self, grid ) :

        P = len( grid )

        def cov_fun( s, t ) : return self.alpha *  \
            np.exp( - self.beta * np.abs( s - t ) )

        return np.array( [ cov_fun( s, t ) for s in grid
                          for t in grid ] ).reshape( P, P )


def generate_gauss_fData( N, mean, Cov ) :

    P = len( Cov.grid )

    assert( len( mean ) == P ), 'You provided mismatching mean and covariance.'

    cholCov = sp.linalg.cholesky( Cov.eval( grid ), lower = False  )

    return np.dot( np.random.normal( 0, 1, N * P ).reshape( N, P ), cholCov ) + mean

    return np.transpose( np.dot( cholCov, \
        np.random.normal( 0, 1, N * P ).reshape( P, N ) ) ) + mean




# Common parameters
# N = 20
P = 1e3
grid = np.linspace( 0, 1, P )

# TESTING the creation of functional basis

# Fourier
# fbasis = FourierBasis( grid, L = 4 )
# fbasis = FourierBasis( grid, ids_subset = [1,3,5,7 ] )
# fbasis = FourierBasis( grid, ids_subset = [2,4,6,8 ] )
# fbasis = FourierBasis( grid, ids_subset = [1,2,3], L = 5 )
# fbasis = FourierBasis( grid, L = 3, ids_subset = [1,3,2] )
# fbasis = FourierBasis( 2 * grid, L = 1, ids_subset = [0] )
# plt.figure()
# [ plt.plot( grid, i ) for i in fbasis.values ]
# plt.show()




# fbasis = BsplineBasis( grid, L = 10 )
# fbasis = BsplineBasis( grid, L = 10, degree = 0 )
# fbasis = BsplineBasis( grid, L = 10, degree = 0, inner_breaks = [0.5, 0.5, 0.7 ] )
# fbasis = BsplineBasis( grid, L = 4, degree = 0, inner_breaks = [0.5, 0.6, 0.7 ] )
# fbasis = BsplineBasis( grid, L = 10, degree = 1 )
fbasis = BsplineBasis( grid, inner_breaks = [ 0.05, 0.2, 0.5, 0.8, 0.95 ] )

plt.figure()
[ plt.plot( grid, i ) for i in fbasis.values ]
plt.grid()
plt.show()

geom = Geom_L2( fbasis )

plt.figure()
plt.imshow( geom.massMatrix().toarray(), interpolation = "nearest" )
plt.colorbar()
plt.show()

# print geom.massMatrix().toarray()


# TESTING the creation of gaussian functional data
# data = np.random.normal( 0, 1, N * P ).reshape( N, P ) + np.sin( 2 * np.pi * grid )
# alpha = 0.4
# beta = 0.5
# Cov = ExpCov( alpha, beta )
# plt.figure()
# plt.imshow( Cov.eval( grid ) )
# plt.colorbar()
# plt.show()


#
# mean = np.sin( 2 * np.pi * grid )
# data = generate_gauss_fData( N, mean, Cov )


#
# fD = fData( grid, data )
# fD.plot()
