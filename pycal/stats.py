import numpy

import scipy as sp
from scipy import linalg, sparse

import copy

# A simple class for functional data objects
class Covariance( object ) :

    def __init__( self, fData ) :

        if( fData.coefs is None ) :
            raise ValueError( 'You provided a functional dataset that has not been expanded on a functional basis')

        self.data_coefs = copy.deepcopy( fData.coefs )

        self.geometry = copy.deepcopy( fData.geometry )

        self._compute()

    def _compute( self ) :

        self.cov_coefs = numpy.cov( self.data_coefs )

    def expand( self ) :

        C = numpy.zeros( ( self.geometry.basis.P ) )

        for i in range( 0, self.geometry.basis.L ) :

            C[ i, ] = numpy.sum( self.geometry.basis.values * self.data_coefs[ i, ][ :, None ], 0 )

    def princomp( self, N_comp = None, threshold = None ) :

        if( threshold is not None ) :

            eigv, eigf = scipy.sparse.linalg.eigs( self.W * self.cov_coefs * self.W, M = self.W, k = self.geometry.basis.L - 1 )

            n_cutoff = numpy.argmax( numpy.cumsum( eigv ** 2 ) / numpy.sum( eigv ** 2 ) >= threshold )

            self.eigenvalues = numpy.delete( eigv, range( n_cutoff + 1, self.geometry.basis.L ) )

            self.eigenfunctions = numpy.delete( eigf, range( n_cutoff + 1,  self.geometry.basis.L ), 1 ).transpose()

        elif( N_comp is not None ) :

                self.eigenvalues, self.eigenvectors = scipy.sparse.linalg.eigs( self.W * self.cov_coefs * self.W, M = self.W, k = N_comp )

                self.eigenvalues = self.eigenvalues.transpose()

        return ( self.eigenvalues, self.eigenfunctions )


    # def plot( self ) :
    #
    #     # @TODO: finish me!
    #
    #     pass

#
# class MedianCovariation( Covariance ) :
#
#     def __init__( self, fData ) :
#
#         Cov.__init__( fData )
#
#         pass
#
#     def _compute( self ) :
#
#
# class SphericalCovariance( Covariance ) :
#
#     def __init__( self, fData ) :
#
#         Cov.__init__( fData )
#
#         pass
#
#     def _compute( self ) :
#
#         pass
