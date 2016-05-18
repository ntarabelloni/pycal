import numpy
import matplotlib.pyplot as plt
import scipy, scipy.sparse

class Geometry( object ) :

    def __init__( self, basis = None ):
        self.basis = basis

        # Mass Matrix
        self.W = None

class Geom_L2( Geometry ) :

    def __init__( self, basis = None ) :

        Geometry.__init__(self, basis )

    def grid_norm( self, x ) :
        z = x * x
        return numpy.sqrt( numpy.sum( ( z[ 1 : ] + z[ : -1 ] ) / 2 * self.basis.h ) )

    def grid_innerProduct( self, x, y ) :
        z = x * y
        return numpy.sum( ( z[ 1 : ] + z[ : - 1 ] ) * 0.5 ) * self.basis.h

    def massMatrix( self ) :

        if( self.W is None ) :

            el = self.basis.values
            L = self.basis.L

            # Lower triangular part of mass matrix
            vals_temp = [ self.grid_innerProduct( el[ i, ], el[ j, ] ) for i in range( 0, L ) for j in range( 0, i ) ]

            vals = [ x for x in vals_temp if abs( x ) >= numpy.finfo(float).eps ]
            ids = [ (i, j) for i in range( 0, L ) for j in range( 0, i )\
                   if abs( vals_temp[ int( i * (i - 1) / 2 ) + j ] ) >= numpy.finfo(float).eps ]

            vals *= 2
            ids += [ (ID[1], ID[0]) for ID in ids ]

            vals += [ self.grid_innerProduct( el[ i, ], el[ i, ] ) for i in range( 0, L ) ]
            ids += [ (i,i) for i in range( 0, L ) ]
            ids = numpy.array( ids )

            self.W = scipy.sparse.csr_matrix( (vals, ( ids[ :, 0], ids[ :, 1 ] )), shape = ( L, L ) )

        return self.W

    def dot( self, x, y ) :
        if( self.W is None ) :
            raise ValueError('Mass matrix is unset. Please, call massMatrix()')
        else :
            return self.W.dot( y ).dot( x )

    def norm( self, x ) :
        if( self.W is None ) :
            raise ValueError('Mass matrix is unset. Please, call massMatrix()')
        else :
            return numpy.sqrt( self.W.dot( x ).dot( x ) )

    def dist( self, x, y ) :
        if( self.W is None ) :
            raise ValueError('Mass matrix is unset. Please, call massMatrix()')
        else :
            return self.norm( x - y )
