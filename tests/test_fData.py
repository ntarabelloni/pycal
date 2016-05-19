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
