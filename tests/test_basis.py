from .context import pycal

import pycal.basis as bs
import unittest
import numpy as np


class FourierBasisTest( unittest.TestCase ) :

    def setUp( self ):
        self.P = 1e3
        self.grid = np.linspace( 0, 1, self. P )
        self.l = np.max( self.grid ) - np.min( self.grid )

    # def tearDown( self ):
    #     pass

    def test_simpleBasis( self ):
        np.testing.assert_array_equal( bs.FourierBasis( self.grid, L = 4 ).values,
                                np.array( [ [1. / np.sqrt( self.l ) ] * int( self.P ), \
                                    np.sin( 2 * np.pi * self.grid ) * np.sqrt( 2  / self.l ), \
                                    np.cos( 2 * np.pi * self.grid ) * np.sqrt( 2  / self.l ), \
                                    np.sin( 4 * np.pi * self.grid ) * np.sqrt( 2  / self.l ) ] ) )

    def test_basisSine( self ) :
        np.testing.assert_array_equal( bs.FourierBasis( self.grid, L = 4, ids_subset = [1,3,5,7] ).values,
                                        np.array( [ np.sin( 2 * np.pi * self.grid ) * np.sqrt( 2 / self.l ), \
                                                    np.sin( 4 * np.pi * self.grid ) * np.sqrt( 2 / self.l ), \
                                                    np.sin( 6 * np.pi * self.grid ) * np.sqrt( 2 / self.l ), \
                                                    np.sin( 8 * np.pi * self.grid ) * np.sqrt( 2 / self.l ) ] ) )

    def test_basisCosine( self ) :
        np.testing.assert_array_equal( bs.FourierBasis( self.grid, L = 4, ids_subset = [2,4,6,8] ).values, \
                                        np.array( [ np.cos( 2 * np.pi * self.grid ) * np.sqrt( 2 / self.l ), \
                                                    np.cos( 4 * np.pi * self.grid ) * np.sqrt( 2 / self.l ), \
                                                    np.cos( 6 * np.pi * self.grid ) * np.sqrt( 2 / self.l ), \
                                                    np.cos( 8 * np.pi * self.grid ) * np.sqrt( 2 / self.l ) ] ) )

    def test_basisStretchGrid( self ) :
        np.testing.assert_array_equal( bs.FourierBasis( 2 * self.grid, L = 4, ids_subset = [0,1,2,5] ).values, \
                                       np.array( [ [ 1. / np.sqrt( 2 * self.l ) ] * int( self.P ),
                                                    np.sin( 2 * np.pi * self.grid ) * np.sqrt( 1 / self.l ), \
                                                    np.cos( 2 * np.pi * self.grid ) * np.sqrt( 1 / self.l ), \
                                                    np.sin( 6 * np.pi * self.grid ) * np.sqrt( 1 / self.l ) ] ) )


    def test_errorUnsortedIds( self ) :
        with self.assertRaises(ValueError):
            bs.FourierBasis( self.grid, L = 3, ids_subset = [1,3,2] )

    def test_errorMismatchingIds( self ) :
        with self.assertRaises(ValueError):
            bs.FourierBasis( self.grid, ids_subset = [1,2,3], L = 5 )

suite = unittest.TestLoader().loadTestsFromTestCase( FourierBasisTest )
unittest.TextTestRunner( verbosity = 2 ).run( suite )



# Common parameters
# N = 20
# P = 1e3
# grid = numpy.linspace( 0, 1, P )

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
# fbasis = BsplineBasis( grid, inner_breaks = [ 0.05, 0.2, 0.5, 0.8, 0.95 ] )

# plt.figure()
# [ plt.plot( grid, i ) for i in fbasis.values ]
# plt.grid()
# plt.show()
#
