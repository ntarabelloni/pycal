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

    def norm( self, x ) :
        z = x * x
        return numpy.sqrt( numpy.sum( ( z[ 1 : ] + z[ : -1 ] ) / 2 * self.basis.h ) )

    def innerProduct( self, x, y ) :
        z = x * y
        return numpy.sum( ( z[ 1 : ] + z[ : - 1 ] ) * 0.5 ) * self.basis.h

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
            ids = numpy.array( ids )

            self.W = scipy.sparse.csr_matrix( (vals, ( ids[ :, 0], ids[ :, 1 ] )), shape = ( L, L ) )

        return self.W
