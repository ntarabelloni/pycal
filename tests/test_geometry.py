from .context import pycal

# Importing modules from pycal
import pycal.basis as bs
import pycal.geometry as geo

# Import testing module
import unittest

# Importing numpy
import numpy as np


class GeometryL2Test( unittest.TestCase ):

    def setUp( self ):
        self.P = 1e2
        self.grid = np.linspace( 0, 1, self.P )
        self.h = self.grid[ 1 ] - self.grid[ 0 ]
        self.tol = 5e-16

    def tearDown( self ):
        pass

    def L2_inner_product( self, x, y ) :
        z = x * y
        return np.sum( ( z[ 1 : ] + z[ : -1 ] ) / 2 * self.h )

    def massMatrix( self, basis ) :
        L = len( basis )
        mass_matrix = np.zeros( ( L, L ) )

        for i in range( 0, L ) :
            mass_matrix[ i, ] = [ self.L2_inner_product( basis[ i, ], basis[j, ] ) \
                                 for j in range( 0, L ) ]
        return mass_matrix

    def test_genericEq( self ) :

        L = 5
        basis = bs.FourierBasis( self.grid, L  )

        G1 = geo.Geom_L2( basis )
        G2 = geo.Geom_L2( basis )

        self.assertTrue( G1 == G2 )

    def test_genericNeq( self ) :

        L1 = 5
        L2 = 4
        B1 = bs.FourierBasis( self.grid, L1  )
        B2 = bs.FourierBasis( self.grid, L2  )

        G1 = geo.Geom_L2( B1 )
        G2 = geo.Geom_L2( B2 )

        self.assertTrue( G1 != G2 )

    def test_genericEqError( self ) :

        L1 = 5
        L2 = 4
        B1 = bs.FourierBasis( self.grid, L1  )
        B2 = bs.FourierBasis( self.grid, L2  )

        G1 = geo.Geom_L2( B1 )
        G2 = geo.Geom_L2( B2 )

        self.assertFalse( G1 == G2 )

    def test_massMatrixFourier_all5( self ):
        L = 5
        fourier = bs.FourierBasis( self.grid, L = L )

        self.assertTrue( np.max( np.abs( self.massMatrix( fourier.values ) - geo.Geom_L2( fourier ).massMatrix().toarray() ) ) < self.tol )

    def test_massMatrixFourier_sine5( self ):
        L = 5
        fourier = bs.FourierBasis( self.grid, L = L, ids_subset = [1,3,5,7,9] )

        self.assertTrue( np.max( np.abs( self.massMatrix( fourier.values ) - geo.Geom_L2( fourier ).massMatrix().toarray() ) ) < self.tol )

    def test_massMatrixFourier_cosine5( self ):
        L = 5
        fourier = bs.FourierBasis( self.grid, L = L, ids_subset = [2,4,6,8,10] )

        self.assertTrue( np.max( np.abs( self.massMatrix( fourier.values ) - geo.Geom_L2( fourier ).massMatrix().toarray() ) ) < self.tol )


    def test_massMatrixBspline_0( self ):

        L = 10
        bspline = bs.BsplineBasis( self.grid, L, degree = 0 )

        self.assertTrue( np.max( np.abs( self.massMatrix( bspline.values ) - \
                                        geo.Geom_L2( bspline ).massMatrix().toarray() ) ) < self.tol )

    def test_massMatrixBspline_1( self ):

        L = 10
        bspline = bs.BsplineBasis( self.grid, L, degree = 1 )

        self.assertTrue( np.max( np.abs( self.massMatrix( bspline.values ) - \
                                        geo.Geom_L2( bspline ).massMatrix().toarray() ) ) < self.tol )

    def test_massMatrixBspline_2( self ):

        L = 10
        bspline = bs.BsplineBasis( self.grid, L, degree = 2 )

        self.assertTrue( np.max( np.abs( self.massMatrix( bspline.values ) - \
                                        geo.Geom_L2( bspline ).massMatrix().toarray() ) ) < self.tol )

    def test_innerProductEqualsNorm( self ):

        G = geo.Geom_L2( bs.FunctionalBasis( self.grid ) )

        x = np.arange( 0, self.P )

        self.assertTrue( np.abs( np.max( np.sqrt( G.grid_innerProduct(  x, x ) ) - G.grid_norm( x ) ) ) < 1e-14 )


    def test_errorDot( self ) :

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        x = [1,2,3,4,5]
        y = [1,2,3,4,5]

        with self.assertRaises( ValueError ) :
            G.dot( x, y )

    def test_errorNorm( self ) :

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        x = [1,2,3,4,5]

        with self.assertRaises( ValueError ) :
            G.norm( x )

    def test_errorDist( self ) :

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        x = [1,2,3,4,5]
        y = [1,2,3,4,5]

        with self.assertRaises( ValueError ) :
            G.dist( x, y )

    def test_norm_dist_dot( self ) :

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        G.massMatrix()

        x = np.array( [1,2,3,4,5] )
        y = np.array( [1,2,3,4,5] )

        self.assertLess( abs( G.norm( x ) - np.sqrt( np.sum( x * x ) ) ), 1e-15 )
        self.assertLess( abs( G.dist( x, y ) ), np.finfo( float ).eps )
        self.assertLess( abs( G.dot( x, y ) - np.sum( x * y ) ), 1e-14 )
