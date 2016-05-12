import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

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

        if( any( numpy.diff( numpy.unique( numpy.diff( grid ) ) ) / \
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

    def plot( self, colors = None, show = False ) :

        if( colors is None ) :
            colors = numpy.random.choice( list( mpl.colors.cnames.keys() ), self.L )

        grid = numpy.linspace( self.t0, self.tP, self.P )
        [ plt.plot( grid, self.values[ i, ], c = colors[ i ] ) for i in range( 0, self.L ) ]
        plt.grid()

        if show :
            plt.show()


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
                    raise ValueError('You provided unsorted ids_subset! ')
                    ids_subset = numpy.sort( ids_subset )
        else :
            ids_subset = range( 0, L )

        # Initialisation of values
        self.L = L
        self.values = numpy.zeros( ( self.L, self.P ) )
        grid = numpy.linspace( self.t0, self.tP, self.P )

        l = numpy.diff( ( numpy.min( grid ), numpy.max( grid ) ) )

        for (pos, i) in zip( range( 0, L ), ids_subset ) :
            if i % 2 == 0 :
                if i == 0 :
                    self.values[ pos ] = 1. / numpy.sqrt( l )
                else :
                    self.values[ pos ] = numpy.sqrt( 2 / l ) * numpy.cos( 2 * numpy.pi * \
                            numpy.ceil( float( i ) / 2 ) * ( grid - grid[0] ) / l )
            else :
                self.values[ pos ] = numpy.sqrt( 2 / l ) * numpy.sin( 2 * numpy.pi * \
                            numpy.ceil( float( i ) / 2 ) * ( grid - grid[ 0 ] ) / l )

class BsplineBasis( FunctionalBasis ) :

    def __init__( self, grid, L = None, degree = 3, inner_breaks = None ) :
        FunctionalBasis.__init__( self, grid )

        self.generateBasis( L, degree, inner_breaks )

    def generateBasis( self, L = None, degree = 3, inner_breaks = None ) :

        self.degree = degree

        grid = numpy.linspace( self.t0, self.tP, self.P )

        if( inner_breaks is None ) :
            if( L is None ) :
                raise ValueError( 'You must provide at least either inner_breaks or L')

            self.L = L
            self.inner_breaks = [ numpy.percentile( grid, prob ) for prob in \
                    numpy.linspace( 0, 100, self.L - 1 - self.degree + 2 )[ 1 : -1 ] ]

            self.values = self.computeBsplines()

        else :
            if( len( inner_breaks ) != len( numpy.unique( inner_breaks ) ) ) :
                raise ValueError( 'Only unique inner breaks are supported for now.')

            if( any( [ inner_breaks[ i ] > inner_breaks[ i + 1 ] \
                      for i in range( 0, len( inner_breaks ) - 1 ) ] ) ) :
                        raise ValueError('You provided unsorted inner_breaks! ')

            if( min( inner_breaks ) <= self.t0 or max( inner_breaks ) >= self.tP ) :
                raise ValueError( 'You provide some inner breaks that fall outside \
                                 or on the boundary of the grid ')
            self.inner_breaks = inner_breaks

            if( L is not None and L != self.degree + 1 + len( self.inner_breaks ) ) :
                raise ValueError( 'L must be equal to degree + 1 + len( inner_breaks )')

            self.L = self.degree + 1 + len( self.inner_breaks )
            self.values = self.computeBsplines()

    def computeBsplines( self ) :

        grid = numpy.linspace( self.t0, self.tP, self.P )

        if( any( [ x >= self.tP or x <= self.t0 for x in self.inner_breaks ] ) ) :
            raise ValueError( 'Inner breaks do not fall inside the grid provided')

        order = self.degree + 1

        N_intervals = len( self.inner_breaks ) + 1

        N = order + len( self.inner_breaks )

        if( isinstance( self.inner_breaks, list ) ) :
            knots_new = numpy.asarray( [ self.t0 ] + self.inner_breaks + [ self.tP ] )
        else :
            # I assume to have a numpy.array or an instance of calss with .tolist()
            # method
            knots_new = numpy.asarray( [ self.t0 ] + self.inner_breaks.tolist() + [ self.tP ] )

        bs_new = numpy.zeros( ( N_intervals, self.P ) )

        for i in range( 0, N_intervals ) :

            bs_new[ i, ] = [ ( x >= knots_new[ i ] ) and \
                            ( True if i == ( len( bs_new ) - 1 )  else x < knots_new[ i + 1 ] ) \
                            for x in grid ]

        if( order == 1 ) :
            return bs_new
        else :
            for K in range( 2, order + 1 ) :

                # The - 1 is due to the 0-based indexing in Python
                offset_old = 1 - ( 3 - K ) - 1
                offset_new = 1 - ( 2 - K ) - 1

                knots_old = knots_new.copy()
                knots_new = numpy.asarray( [ self.t0 ] + knots_old.tolist() + [ self.tP ] )

                bs_old = bs_new.copy()
                bs_new = numpy.zeros( ( N_intervals + K - 1, self.P ) )

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
