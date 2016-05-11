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
                                np.array( [ [1. / self.l ] * int( self.P ), \
                                    np.sin( 2 * np.pi * self.grid ) * np.sqrt( 2  / self.l ), \
                                    np.cos( 2 * np.pi * self.grid ) * np.sqrt( 2  / self.l ), \
                                    np.sin( 4 * np.pi * self.grid ) * np.sqrt( 2  / self.l ) ] ) )

    def test_errorUnsortedIds( self ) :
        with self.assertRaises(ValueError):
            bs.FourierBasis( self.grid, L = 3, ids_subset = [1,3,2] )

    def test_errorMismatchingIds( self ) :
        with self.assertRaises(ValueError):
            bs.FourierBasis( self.grid, ids_subset = [1,2,3], L = 5 )

suite = unittest.TestLoader().loadTestsFromTestCase( FourierBasisTest )
unittest.TextTestRunner( verbosity = 2 ).run( suite )

    #
    # for (pos, i) in zip( range( 0, L ), ids_subset ) :
    #     if i % 2 == 0 :
    #         if i == 0 :
    #             self.values[ pos ] = 1. / numpy.sqrt( l )
    #         else :
    #             self.values[ pos ] = numpy.sqrt( 2 / l ) * numpy.cos( 2 * numpy.pi * \
    #                     numpy.ceil( float( i ) / 2 ) * ( grid - grid[0] ) / l )
    #     else :
    #         self.values[ pos ] = numpy.sqrt( 2 / l ) * numpy.sin( 2 * numpy.pi * \
    #                     numpy.ceil( float( i ) / 2 ) * ( grid - grid[ 0 ] ) / l )
    #
    #



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
