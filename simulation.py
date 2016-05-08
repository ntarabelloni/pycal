import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy import linalg, sparse


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
