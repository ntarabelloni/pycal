import numpy

import scipy
from scipy import linalg, sparse

import copy

class Covariance( object ) :

    def __init__( self, fData ) :

        if( fData.coefs is None ) :
            raise ValueError( 'You provided a functional dataset that has not been expanded on a functional basis')

        self.data_coefs = copy.deepcopy( fData.coefs )

        self.geometry = copy.deepcopy( fData.geometry )

        self._compute()

        self.eigenv = None
        self.eigenf = None
        self.eigenf_coefs = None

    def _compute( self ) :

        self.coefs = numpy.cov( self.data_coefs, rowvar = False )

    def expand( self ) :

        return  numpy.dot( self.geometry.basis.values.transpose(), \
                          numpy.dot( self.coefs, \
                                     self.geometry.basis.values ) )

    def _explainedVarianceRatio( self, v ) :

        return numpy.cumsum( v[ 0 : len( v ) ]**2 ) / numpy.sum( v**2 )


    def princomp( self, N_comp = None, threshold = None ) :

        if( threshold is not None ) :

            eigv, eigf = scipy.sparse.linalg.eigsh( self.geometry.W * self.coefs * self.geometry.W, M = self.geometry.W, k = 10, which = 'LM' )

            if ( self._explainedVarianceRatio( eigv )[ -1 ] < threshold ) :

                eigv, eigf = scipy.sparse.linalg.eigs( self.geometry.W * self.coefs * self.geometry.W, \
                                                      M = self.geometry.W, \
                                                      k = self.geometry.basis.L - 1, \
                                                      which = 'LM' )

            # Eigenvalues are sorted in increasing order, but I want them in decreasing
            eigv = eigv[ ::-1 ]

            k = len( eigv )

            n_cutoff = numpy.argmax( self._explainedVarianceRatio( eigv ) >= threshold ) + 1

            self.eigenv = eigv[ 0 : n_cutoff ]

            #  And eigenfunctions accordingly
            self.eigenf_coefs = eigf[ :, range( k - 1, k - 1 - n_cutoff, -1 ) ].transpose()

        elif( N_comp is not None ) :

                self.eigenv, self.eigenf_coefs = \
                scipy.sparse.linalg.eigsh( self.geometry.W * self.coefs * self.geometry.W, \
                                            M = self.geometry.W, \
                                            k = N_comp, \
                                            which = 'LM' )

                #  Eigenvalues are sorted in increasing order, but I want them in decreasing
                self.eigenv = self.eigenv[ ::-1 ]

                #  And eigenfunctions accordingly
                self.eigenf_coefs = self.eigenf_coefs[ :, ::-1 ].transpose()

        else :
            raise ValueError( 'Please specify one between N_comp and threshold' )

        self.eigenf = numpy.dot( self.eigenf_coefs, self.geometry.basis.values  )

        return ( self.eigenv, self.eigenf )


    # def plot( self ) :
    #
    #     # @TODO: finish me!
    #
    #     pass


class MedianCovariation( Covariance ) :

    def __init__( self, fData ) :

        Cov.__init__( fData )

        pass

    def _compute( self ) :
        pass

class SphericalCovariance( Covariance ) :

    def __init__( self, fData ) :

        Cov.__init__( fData )

        pass

    def _compute( self ) :

        pass
