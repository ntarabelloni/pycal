from .context import pycal

import pycal.basis as bs
import unittest
import numpy as np


class FourierBasisTest( unittest.TestCase ) :

    def setUp( self ):
        self.P = 1e3
        self.grid = np.linspace( 0, 1, self. P )
        self.l = np.max( self.grid ) - np.min( self.grid )

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
        with self.assertRaises( ValueError ):
            bs.FourierBasis( self.grid, L = 3, ids_subset = [1,3,2] )

    def test_errorMismatchingIds( self ) :
        with self.assertRaises( ValueError ):
            bs.FourierBasis( self.grid, ids_subset = [1,2,3], L = 5 )


# suite = unittest.TestLoader().loadTestsFromTestCase( FourierBasisTest )
# unittest.TextTestRunner( verbosity = 2 ).run( suite )


class BsplineBasisTest( unittest.TestCase ) :

    def setUp( self ):
        self.P = 101
        self.grid = np.linspace( 0, 1, self.P )

    # Testing the generation of a 0-order basis on uniform knots
    def test_zeroOrderUniformKnots( self ) :

        L = 4
        vals = np.zeros( ( L, self.P ) )

        inner_breaks = [ 0.25, 0.5, 0.75 ]
        knots = [0.] + inner_breaks + [1.]
        for i in range( 0, L ) :
            vals[ i, ] = ( self.grid >= knots[ i ] ) * 1. *  \
                         ( self.grid < knots[ i + 1 ] ) * 1.

        vals[ -1, -1 ] = 1.

        np.testing.assert_array_equal( bs.BsplineBasis( self.grid, \
                                                       L, 0, \
                                                       inner_breaks ).values,
                                          vals )

    # Testing the generation of a 0-order basis on non-uniform knots
    def test_zeroOrderNonUniformKnots( self ) :

        L = 4
        vals = np.zeros( ( L, self.P ) )

        inner_breaks = [ 0.3, 0.4, 0.9]
        knots = [0.] + inner_breaks + [1.]

        for i in range( 0, L ) :
            vals[ i, ] = ( self.grid >= knots[ i ] ) * 1. *  \
                         ( self.grid < knots[ i + 1 ] ) * 1.

        vals[ -1, -1 ] = 1.

        np.testing.assert_array_equal( bs.BsplineBasis( self.grid, \
                                                        L, 0, \
                                                        inner_breaks ).values,
                                                        vals )


    def test_firstOrderUniformKnots( self ) :
        np.testing.assert_array_equal( bs.BsplineBasis( self.grid, 5, 1, [ 0.25, 0.5, 0.75 ] ).values,
                                       self.firstOrderValues( self.grid, [ 0., 0.25, 0.5, 0.75, 1. ] ) )

    def test_firstOrderNonUniformKnots( self ) :
        self.assertTrue( np.sum( abs( self.firstOrderValues( self.grid, [0., 0.3, 0.4, 0.8, 1.] ) - \
                            bs.BsplineBasis( self.grid, 5, 1, [ 0.3, 0.4, 0.8 ] ).values )  ) < 3e-15 )

    def test_errorMismatchingParams1( self ) :
        with self.assertRaises( ValueError ) :
            bs.BsplineBasis( self.grid, L = 3, degree = 1, inner_breaks = [0.25, 0.5, 0.75 ] )

    def test_errorMismatchingParams2( self ) :
            with self.assertRaises( ValueError ) :
                bs.BsplineBasis( self.grid, inner_breaks = [0.5, 0.25, 0.75 ] )

    def firstOrderValues( self, grid, knots ) :

        K = len( knots )
        P = len( grid )

        values = np.zeros( ( K, P ) )

        f_ascending = lambda x, x0, x1 : ( x - x0 ) / ( x1 - x0 )
        f_descending = lambda x, x0, x1 : 1 - ( x - x0 ) / ( x1 - x0 )

        for i in range( 0, K - 1 ) :

            ids = [ idx for ( idx, x ) in zip( range( 0, P), grid ) if x <= knots[ i + 1 ] and x >= knots[ i ] ]

            values[ i + 1, ids[ 1 : -1 ] ] += f_ascending( grid[ ids[ 1 : -1 ] ], knots[ i ], knots[ i + 1 ] )
            values[ i, ids ] += f_descending( grid[ ids ], knots[ i ], knots[ i + 1 ] )

        values[ K - 1 , - 1 ] = 1.

        return values



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
