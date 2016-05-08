import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy import linalg, sparse

import basis as bs
import geometry as geo
import simulation as sim

# Common parameters
# N = 20
P = 1e3
grid = np.linspace( 0, 1, P )

# TESTING the creation of functional basis

# # Fourier
# fbasis = bs.FourierBasis( grid, L = 4 )
# fbasis = bs.FourierBasis( grid, L = 4, ids_subset = [1,3,5,7 ] )
# fbasis = bs.FourierBasis( grid, L = 4, ids_subset = [2,4,6,8 ] )
# fbasis = bs.FourierBasis( grid, ids_subset = [1,2,3], L = 5 )
# fbasis = bs.FourierBasis( grid, L = 3, ids_subset = [1,3,2] )
# fbasis = bs.FourierBasis( 2 * grid, L = 1, ids_subset = [0] )
# plt.figure()
# [ plt.plot( grid, i ) for i in fbasis.values ]
# plt.show()




fbasis = bs.BsplineBasis( grid, L = 10 )
fbasis = bs.BsplineBasis( grid, L = 10, degree = 0 )
# fbasis = bs.BsplineBasis( grid, L = 10, degree = 0, inner_breaks = [0.5, 0.5, 0.7 ] )
fbasis = bs.BsplineBasis( grid, L = 4, degree = 0, inner_breaks = [0.5, 0.6, 0.7 ] )
fbasis = bs.BsplineBasis( grid, L = 10, degree = 1 )

fbasis = bs.BsplineBasis( grid, inner_breaks = [ 0.05, 0.2, 0.5, 0.8, 0.95 ] )

plt.figure()
[ plt.plot( grid, i ) for i in fbasis.values ]
plt.grid()
plt.show()

Geo = geo.Geom_L2( fbasis )

plt.figure()
plt.imshow( Geo.massMatrix().toarray(), interpolation = "nearest" )
plt.colorbar()
plt.show()

# print geom.massMatrix().toarray()


# TESTING the creation of gaussian functional data
# data = np.random.normal( 0, 1, N * P ).reshape( N, P ) + np.sin( 2 * np.pi * grid )
# alpha = 0.4
# beta = 0.5
# Cov = ExpCov( alpha, beta )
# plt.figure()
# plt.imshow( Cov.eval( grid ) )
# plt.colorbar()
# plt.show()


#
# mean = np.sin( 2 * np.pi * grid )
# data = generate_gauss_fData( N, mean, Cov )


#
# fD = fData( grid, data )
# fD.plot()
