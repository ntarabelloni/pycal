import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy import linalg, sparse

# A simple class for functional data objects
class fData(object) :
    def __init__( self, grid = None, data = None ) :
        self.grid = grid.copy()
        self.data = data.copy()

    def plot( self ) :
        plt.figure()
        [ plt.plot( self.grid, self.data[ i, : ]) for i in range( 0, N ) ]
        plt.show()

# A simple class that implements an Exponential Covariance function
class ExpCov(object) :

    def __init__( self, alpha, beta ) :
        self.alpha = alpha
        self.beta = beta

    def eval( self, grid ) :

        P = len( grid )

        def cov_fun( s, t ) : return self.alpha *  \
            np.exp( - self.beta * np.abs( s - t ) )

        return np.array( [ cov_fun( s, t ) for s in grid
                          for t in grid ] ).reshape( P, P )


def generate_gauss_fData( N, mean, Cov ) :

    P = len( Cov.grid )

    assert( len( mean ) == P ), 'You provided mismatching mean and covariance.'

    cholCov = sp.linalg.cholesky( Cov.eval( grid ), lower = False  )

    return np.dot( np.random.normal( 0, 1, N * P ).reshape( N, P ), cholCov ) + mean

    return np.transpose( np.dot( cholCov, \
        np.random.normal( 0, 1, N * P ).reshape( P, N ) ) ) + mean


class FunctionalBasis( object ) :

    # Default constructor of class
    def __init__( self, grid = None, values = None ) :

        # Grid parameters
        self.t0 = None
        self.tP = None
        self.h = None
        self.P = None

        # Basis parameters
        self.values = None
        self.L = None

        if( grid is not None ) :
            self.setGrid( grid )

        if( values is not None ) :
            self.setValues( values )

    def setGrid( self, grid ) :

        if( self.values is not None and len( grid ) != self.values.shape[1] ) :
            raise ValueError( 'You provided a grid which is not compliant to \
                             the basis values')

        if( any( np.diff( np.unique( np.diff( grid ) ) ) / \
            ( grid.max() - grid.min() ) > 1e-14 ) ) :
                raise ValueError( 'You provided a non-uniformly spaced grid' )

        self.t0 = grid[ 0 ]
        self.tP = grid[ - 1 ]
        self.h = grid[ 1 ] - grid[ 0 ]
        self.P = len( grid )


    def setValues( self, values ) :

        if( self.t0 is not None ) :

            if( self.tP is None or self.h is None or self.P is None ) :
                raise ValueError('Some grid parameters are unexpectedly set to None')

            if( self.P != values.shape[1] ) :
                raise ValueError( 'You provided values discretised on a \
                                 mismatching grid than the stored one')

        self.L = len( values )
        self.values = values.copy()


class FourierBasis( FunctionalBasis ) :

    def __init__( self, grid, L, ids_subset = None ) :
        FunctionalBasis.__init__( self, grid )

        self.generateBasis( L, ids_subset )

    def generateBasis( self, L, ids_subset) :

        # It generates the fourier basis elements of type:
        #
        #    sin( ( k * pi * grid ) if ( k % 2 == 1 )
        #    cos( k / 2 * pi * grid ) if ( k % 2 == 0 )

        if( L is None and ids_subset is None ) :
            raise ValueError( 'Please, provide at leas one between L and ids_subset')
        elif( not ids_subset is None ) :
            if( not L is None and L != len( ids_subset ) ) :
                raise ValueError( 'You provided mismatching L and ids_subset' )
            else :
                L = len( ids_subset )
                if( any( [ ids_subset[ i ] > ids_subset[ i + 1 ] \
                          for i in range( 0, L - 1 ) ] ) ) :
                    raise UserWarning('You provided unsorted ids_subset! Sorting it. ')
                    ids_subset = np.sort( ids_subset )
        else :
            ids_subset = range( 0, L )

        # Initialisation of values
        self.L = L
        self.values = np.zeros( ( self.L, self.P ) )
        grid = np.linspace( self.t0, self.tP, self.P )

        l = np.diff( ( np.min( grid ), np.max( grid ) ) )
        print l

        for (pos, i) in zip( range( 0, L ), ids_subset ) :
            if i % 2 == 0 :
                if i == 0 :
                    self.values[ pos ] = 1. / np.sqrt( l )
                else :
                    self.values[ pos ] = np.sqrt( 2 / l ) * np.cos( 2 * np.pi * \
                            np.ceil( float( i ) / 2 ) * ( grid - grid[0] ) / l )
            else :
                self.values[ pos ] = np.sqrt( 2 / l ) * np.sin( 2 * np.pi * \
                            np.ceil( float( i ) / 2 ) * ( grid - grid[ 0 ] ) / l )

class BsplineBasis( FunctionalBasis ) :

    def __init__( self, grid, L = None, degree = 3, inner_breaks = None ) :
        FunctionalBasis.__init__( self, grid )

        self.generateBasis( L, degree, inner_breaks )

    def generateBasis( self, L = None, degree = 3, inner_breaks = None ) :

        self.degree = degree

        grid = np.linspace( self.t0, self.tP, self.P )

        if( inner_breaks is None ) :
            if( L is None ) :
                raise ValueError( 'You must provide at least either inner_breaks or L')

            self.L = L
            self.inner_breaks = [ np.percentile( grid, prob ) for prob in \
                    np.linspace( 0, 100, self.L - 1 - self.degree + 2 )[ 1 : -1 ] ]

            self.values = self.computeBsplines()

        else :
            if( len( inner_breaks ) != len( np.unique( inner_breaks ) ) ) :
                raise ValueError( 'Only unique inner breaks are supported for now.')

            if( min( inner_breaks ) <= self.t0 or max( inner_breaks ) >= self.tP ) :
                raise ValueError( 'You provide some inner breaks that fall outside \
                                 or on the boundary of the grid ')
            self.inner_breaks = inner_breaks

            if( L is not None and L != self.degree + 1 + len( self.inner_breaks ) ) :
                raise ValueError( 'L must be equal to degree + 1 + len( inner_breaks )')

            self.L = self.degree + 1 + len( self.inner_breaks )
            self.values = self.computeBsplines()

    def computeBsplines( self ) :

        grid = np.linspace( self.t0, self.tP, self.P )

        if( any( [ x >= self.tP or x <= self.t0 for x in self.inner_breaks ] ) ) :
            raise ValueError( 'Inner breaks do not fall inside the grid provided')

        order = self.degree + 1

        N_intervals = len( self.inner_breaks ) + 1

        N = order + len( self.inner_breaks )

        if( isinstance( self.inner_breaks, list ) ) :
            knots_new = np.asarray( [ self.t0 ] + self.inner_breaks + [ self.tP ] )
        else :
            # I assume to have a numpy.array or an instance of calss with .tolist()
            # method
            knots_new = np.asarray( [ self.t0 ] + self.inner_breaks.tolist() + [ self.tP ] )

        bs_new = np.zeros( ( N_intervals, self.P ) )

        for i in range( 0, N_intervals ) :

            bs_new[ i, ] = [ ( x >= knots_new[ i ] ) and \
                            ( True if i == len( bs_new ) else x < knots_new[ i + 1 ] ) \
                            for x in grid ]

        if( order == 1 ) :
            return bs_new
        else :
            for K in range( 2, order + 1 ) :

                # The - 1 is due to the 0-based indexing in Python
                offset_old = 1 - ( 3 - K ) - 1
                offset_new = 1 - ( 2 - K ) - 1

                knots_old = knots_new.copy()
                knots_new = np.asarray( [ self.t0 ] + knots_old.tolist() + [ self.tP ] )

                bs_old = bs_new.copy()
                bs_new = np.zeros( ( N_intervals + K - 1, self.P ) )

                for i in range( 2 - K, N_intervals + 1 ) :

                    # To understand this, make a "tree" diagram of B1,1 ... B4,1
                    # and update up to the third layer
                    if( i >= ( 3 - K ) ) :
                        bs_new[ offset_new + i, ] = ( grid - knots_new[ offset_new + i ] ) / \
                            ( knots_new[ offset_new + i + K - 1 ] - \
                              knots_new[ offset_new + i ] ) * \
                            bs_old[ offset_old + i, ]

                    if( i + 1 <= N_intervals ) :
                        bs_new[ offset_new + i, ] = bs_new[ offset_new + i, ] + \
                            ( knots_new[ offset_new + i + K ] - grid ) / \
                            ( knots_new[ offset_new + i + K ] - knots_new[ offset_new + i + 1 ] ) * \
                                bs_old[ offset_old + i + 1, ]
                K += 1

        return bs_new



class Geometry( object ) :

    def __init__( self, basis = None ):
        self.basis = basis

        # Mass Matrix
        self.W = None

class Geom_L2( Geometry ) :

    def __init__( self, basis = None ) :

        Geometry.__init__(self, basis )

    def norm( self, x ) :
        z = x * x
        return np.sqrt( np.sum( ( z[ 1 : ] + z[ : -1 ] ) / 2 * self.basis.h ) )

    def innerProduct( self, x, y ) :
        z = x * y
        return np.sum( ( z[ 1 : ] + z[ : - 1 ] ) * 0.5 ) * self.basis.h

    def massMatrix( self ) :

        if( self.W is None ) :

            el = self.basis.values
            L = self.basis.L

            # Lower triangular part of mass matrix
            vals = [ self.innerProduct( el[ i, ], el[ j, ] ) for i in range( 0, L ) for j in range( 0, i ) ]
            vals = vals * 2
            vals += [ self.innerProduct( el[ i, ], el[ i, ] ) for i in range( 0, L ) ]

            ids = [ (i,j) for i in range( 0, L ) for j in range( 0, i ) ]
            ids += [ (j,i) for i in range( 0, L ) for j in range( 0, i ) ]
            ids += [ (i,i) for i in range( 0, L ) ]
            ids = np.array( ids )

            self.W = sp.sparse.csr_matrix( (vals, ( ids[ :, 0], ids[ :, 1 ] )), shape = ( L, L ) )

        return self.W


# Common parameters
# N = 20
P = 1e3
grid = np.linspace( 0, 1, P )

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
fbasis = BsplineBasis( grid, inner_breaks = [ 0.05, 0.2, 0.5, 0.8, 0.95 ] )

plt.figure()
[ plt.plot( grid, i ) for i in fbasis.values ]
plt.grid()
plt.show()

geom = Geom_L2( fbasis )

plt.figure()
plt.imshow( geom.massMatrix().toarray(), interpolation = "nearest" )
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
