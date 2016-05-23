from .context import pycal

# Importing modules from pycal
import pycal.basis as bs
import pycal.geometry as geo
import pycal.fData as fd
import pycal.simulation as sim
import pycal.stats as st

# Import testing module
import unittest

# Importing numpy
import numpy as np

import copy


class CovarianceTest( unittest.TestCase ) :

    def setUp( self ) :

        self.P = 100
        self.N = 30
        self.grid = np.linspace( 0, 1, self.P )

        data = sim.generate_gauss_fData( self.N,\
                                        np.sin( 2 * np.pi * self.grid ), \
                                        sim.ExpCov( 0.4, 0.5 ).eval( self.grid ) )

        self.fD = fd.fData( data, self.grid )

    def test_build( self ) :

        G1 = geo.Geom_L2( bs.FourierBasis( self.grid, L = 10 ) )
        G2 = geo.Geom_L2( bs.BsplineBasis( self.grid, L = 10, degree = 3 ) )

        self.fD.addGeometry( G1 )

        F1 = self.fD
        F2 = copy.deepcopy( self.fD )

        F2.addGeometry( G2 )

        C1 = st.Covariance( F1 )
        C2 = st.Covariance( F2 )

        self.assertTrue( C1.geometry == G1 )
        self.assertTrue( C2.geometry == G2 )

        np.testing.assert_array_equal( C1.data_coefs, F1.coefs )
        np.testing.assert_array_equal( C2.data_coefs, F2.coefs )
