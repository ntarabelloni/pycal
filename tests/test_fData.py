from .context import pycal

# Importing modules from pycal
import pycal.basis as bs
import pycal.simulation as sim
import pycal.fData as fd
import pycal.geometry as geo

# Import testing module
import unittest

# Importing numpy
import numpy as np


class FDataTest( unittest.TestCase ) :

    def setUp( self ) :
        # Parameters to define the grid for univariate functional data
        self.t0 = 0
        self.tP = 1
        self.P = 1e2

        self.grid = np.linspace( self.t0, self.tP, self.P )

        # Generation of simulated functional data
        self.N = 1e2
        mean = np.sin( 2 * np.pi * self.grid )
        alpha = 0.2
        beta = 0.4

        self.data = sim.generate_gauss_fData( self.N, mean, \
                                             sim.ExpCov( alpha, beta ).eval( self.grid ) )

    def test_pointwiseBuild( self ) :

        F = fd.fData( self.data, self.grid  )

        np.testing.assert_array_equal( F.data, self.data )
        np.testing.assert_array_equal( F.grid, self.grid )

        self.assertTrue( F.coefs is None )
        self.assertTrue( F.geometry is None )
        self.assertTrue( F.coefs is None )

    def test_geometryBuild( self ) :

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F = fd.fData( self.data, geometry = G )

        np.testing.assert_array_equal( self.grid, F.grid )
        np.testing.assert_array_equal( self.data, F.data )
        self.assertTrue( F.coefs is not None )

    def test_addGeometry( self ) :

        F = fd.fData( self.data, self.grid )

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F.addGeometry( G )

        np.testing.assert_array_equal( self.grid, F.grid )
        np.testing.assert_array_equal( self.data, F.data )
        self.assertTrue( F.coefs is not None )

    def test_errorGrid( self ) :

        F = fd.fData( self.data, self.grid )

        G = geo.Geom_L2( bs.FourierBasis( 2 * self.grid, L = 5 ) )

        with self.assertRaises( ValueError ) :
            F.addGeometry( G )

    def test_errorData( self ) :

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        with self.assertRaises( ValueError ) :
            fd.fData( grid = self.grid, geometry = G )

        with self.assertRaises( ValueError ) :
            fd.fData( geometry = G )

    def test_setGrid( self ) :

        F1 = fd.fData( self.data )

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F2 = fd.fData( self.data, geometry = G )

        F1.setGrid( self.grid )
        F2.setGrid( self.grid )

        np.testing.assert_array_equal( self.grid, F1.grid )
        np.testing.assert_array_equal( self.grid, F2.grid )


    def test_setGridError( self ) :

        F1 = fd.fData( self.data )

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F2 = fd.fData( self.data, geometry = G )

        with self.assertRaises( ValueError ) :
            F1.setGrid( np.delete( self.grid, 0 ) )
        with self.assertRaises( ValueError ) :
            F1.setGrid( np.delete( self.grid, 1 ) )
        with self.assertRaises( ValueError ) :
            F1.setGrid( np.delete( self.grid, -1 ) )

        with self.assertRaises( ValueError ) :
            F2.setGrid( np.delete( self.grid, 0 ) )
        with self.assertRaises( ValueError ) :
            F2.setGrid( np.delete( self.grid, 1 ) )
        with self.assertRaises( ValueError ) :
            F2.setGrid( np.delete( self.grid, -1 ) )

    def test_setData( self ) :

        F1 = fd.fData( np.delete( self.data, -1, 0 ), self.grid )

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F2 = fd.fData( np.delete( self.data, -1 , 0 ), self.grid, G )

        F1.setData( self.data )
        F2.setData( self.data )

        np.testing.assert_array_equal( F1.data, self.data )
        np.testing.assert_array_equal( F2.data, self.data )
        self.assertTrue( F2.coefs is not None )

    def test_setDataError( self ) :

        F1 = fd.fData( grid = self.grid )

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F2 = fd.fData( self.data, self.grid, G )

        with self.assertRaises( ValueError ) :
            F1.setData( np.delete( self.data, 1, 1 ) )

        with self.assertRaises( ValueError ) :
            F2.setData( np.delete( self.data, 1, 1 ) )

    def test_pop( self ) :

        F1 = fd.fData( self.data, self.grid )
        F2 = fd.fData( self.data, self.grid )

        F1.pop( 1 )
        F2.pop( range( 0, 3 ) )

        np.testing.assert_array_equal( F1.data, np.delete( self.data, 1, 0 ) )
        np.testing.assert_array_equal( F2.data, np.delete( self.data, range( 0, 3 ) , 0 ) )

        self.assertEqual( F1.N, self.N - 1 )
        self.assertEqual( F2.N, self.N - 3 )

    def test_str( self ) :

        F = fd.fData( self.data, self.grid )

        self.assertEqual( str( F ), ' Functional Dataset with shape ' + str( self.data.shape ) )

    def test_addGeometry( self ) :

        F1 = fd.fData( self.data )
        F2 = fd.fData( self.data, self.grid )

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F1.addGeometry( G )
        F2.addGeometry( G )

        np.testing.assert_array_equal( F1.grid, self.grid )
        np.testing.assert_array_equal( F2.grid, self.grid )

        self.assertTrue( F1.coefs is not None )
        self.assertTrue( F1.coefs is not None )

        self.assertEqual( F1.geometry, G )
        self.assertEqual( F2.geometry, G )


    def test_addGeometryError( self ) :

        F1 = fd.fData( self.data )
        F2 = fd.fData( self.data, self.grid )

        G = geo.Geom_L2( bs.FourierBasis( np.delete( self.grid, 0 ), L = 5 ) )

        with self.assertRaises( ValueError ) :
            F1.addGeometry( G )

        with self.assertRaises( ValueError ) :
            F2.addGeometry( G )

    def test_expand( self ) :

        D = np.array( [ np.sin( 2 * np.pi * self.grid ), \
                          np.cos( 2 * np.pi * self.grid ), \
                          [1.] * int( self.P ) ] )

        G = geo.Geom_L2( bs.FourierBasis( self.grid, L = 5 ) )

        F1 = fd.fData( D, self.grid, G )

        self.assertLess( np.sum( abs( F1.expand() - D ) ), 1e-13 )
