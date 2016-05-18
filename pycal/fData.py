import numpy
import matplotlib.pyplot as plt

import scipy, scipy.linalg

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
            numpy.exp( - self.beta * numpy.abs( s - t ) )

        return numpy.array( [ cov_fun( s, t ) for s in grid
                          for t in grid ] ).reshape( P, P )


def generate_gauss_fData( N, mean, Cov ) :

    P = len( Cov.grid )

    assert( len( mean ) == P ), 'You provided mismatching mean and covariance.'

    cholCov = scipy.linalg.cholesky( Cov.eval( grid ), lower = False  )

    return numpy.dot( numpy.random.normal( 0, 1, N * P ).reshape( N, P ), cholCov ) + mean

    return numpy.transpose( numpy.dot( cholCov, \
        numpy.random.normal( 0, 1, N * P ).reshape( P, N ) ) ) + mean





# geom = Geom_L2( fbasis )
#
# plt.figure()
# plt.imshow( geom.massMatrix().toarray(), interpolation = "nearest" )
# plt.colorbar()
# plt.show()

# print geom.massMatrix().toarray()


# TESTING the creation of gaussian functional data
# data = numpy.random.normal( 0, 1, N * P ).reshape( N, P ) + numpy.sin( 2 * numpy.pi * grid )
# alpha = 0.4
# beta = 0.5
# Cov = ExpCov( alpha, beta )
# plt.figure()
# plt.imshow( Cov.eval( grid ) )
# plt.colorbar()
# plt.show()


#
# mean = numpy.sin( 2 * numpy.pi * grid )
# data = generate_gauss_fData( N, mean, Cov )


#
# fD = fData( grid, data )
# fD.plot()
