import numpy
import matplotlib.pyplot as plt

import scipy, scipy.linalg, scipy.sparse, scipy.sparse.linalg

# A simple class for functional data objects
class fData(object) :

    def __init__( self, data = None, grid = None, geometry = None ) :
        self.data = data.copy() if data is not None else None
        self.N = len( data ) if data is not None else None

        self.geometry = geometry
        self.grid = grid

        self.coefs = None


        if ( self.grid is not None and self.data is not None ) :
            if( len( self.grid ) != self.data.shape[1] ) :
                raise ValueError('You provided a grid which is not compliant with data')

        if( geometry is not None ) :

            # Reference to shorten commands
            bs = self.geometry.basis

            if( self.grid is not None ) :
                if( self.grid[ 0 ] != bs.t0 or self.grid[ -1 ] != bs.tP or \
                    len( self.grid ) != bs.P or self.grid[ 1 ] - self.grid[ 0 ] != bs.h ) :
                    raise ValueError('The grid you provided is not compliant with that of the geometry')
            else :
                self.grid = numpy.linspace( bs.t0, bs.tP, bs.P )

            if( data is None ) :
                raise ValueError('You provided a geometry but not a dataset')
            else :
                self._project()

    def setGrid( self, grid ) :

        self.grid = grid

        if( self.data is not None ) :
            if( self.data.shape[ 1 ] != len( self.grid ) ) :
                raise ValueError('The grid you provided is not compliant with data')

        if( self.geometry is not None ) :

            bs = self.geometry.basis

            if( self.grid[ 0 ] != bs.t0 or self.grid[ -1 ] != bs.tP or \
                len( self.grid ) != bs.P or self.grid[ 1 ] - self.grid[ 0 ] != bs.h ) :
                raise ValueError('The grid you provided is not compliant with that of the geometry')

        return

    def setData( self, data ) :

        if( self.grid is not None ) :

            if( len( self.grid ) != data.shape[ 1 ] ) :
                raise ValueError( 'The grid is not compliant with the provided dataset' )

        self.data = data.copy()

        if( self.geometry is not None ) :

            self._project()

        return

    def _project( self ) :

        rhs = self._computeProjectionRHS()

        self.coefs = numpy.zeros( ( self.N, self.geometry.basis.L ) )

        conv_flag = 0

        for i in range( 0, self.N ) :

            # Solving the linear problem
            res = scipy.sparse.linalg.cg( self.geometry.massMatrix(), rhs[ :, i ] )

            conv_flag += res[ 1 ]
            self.coefs[ i, ] = res[ 0 ]

        if( res[1] > 0 ) :
            raise ValueError('CG did not converge to the solution')


    def _computeProjectionRHS( self ) :

        rhs = numpy.zeros( ( self.geometry.basis.L, self.N ) )

        # Reference to shorten commands
        el = self.geometry.basis.values

        for i in range( 0, self.N ) :
            rhs[ : , i ] = [ self.geometry.grid_innerProduct( self.data[ i, ], el[ j, ] ) \
                         for j in range( 0, self.geometry.basis.L ) ]
        return rhs


    def expand( self ) :

        if( self.coefs is None or self.geometry is None ) :
            raise ValueError('Geometry not proivided or data not projected yet')

        values = numpy.zeros( ( self.N, self.geometry.basis.P ) )

        for i in range( 0, self.N ) :
            # The [ :, None ] is required to cast numpy ndarrays from (L,) into (L,1)
            values[ i, ] = numpy.sum( self.geometry.basis.values * self.coefs[ i, ][ :, None ], 0 )

        return values

    def addGeometry( self, geometry ) :

        # Reference to shorten commands
        bs = geometry.basis

        if( self.grid is not None ) :
            if( self.grid[ 0 ] != bs.t0 or self.grid[ -1 ] != bs.tP or \
                len( self.grid ) != bs.P or self.grid[ 1 ] - self.grid[ 0 ] != bs.h ) :
                raise ValueError('The grid you provided is not compliant with that of the geometry')
        else :
            self.grid = numpy.linspace( bs.t0, bs.tP, bs.P )

        self.geometry = geometry

        # Calling the projection method
        self._project()

    # Override of selection operator
    def __getitem__( self, key ) :

        #@NOTE: Finish me!

        pass

    def pop( self, key ) :

        self.data = numpy.delete( self.data, key, axis = 0 )

        self.N = len( self.data )

        if ( self.coefs is not None ) :

            self.coefs = numpy.delete( self.coefs, key, axis = 0 )


    def __str__( self ) :

        descr =  ' Functional Dataset with shape ' + str( self.data.shape )

        return descr

    def plot( self ) :
        plt.figure()
        [ plt.plot( self.grid, self.data[ i, : ]) for i in range( 0, N ) ]
        plt.show()
