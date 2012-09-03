import numpy
from constants import DTYPE,INTDTYPE

class FixedMesh(object):  
    'common parts of TriMesh and QuadMesh'
    clsname = 'FixedMesh'
    verts_per_face = None
    verts_per_face_name = None
    def __init__(self,
                 points,
                 indicies,
                 name   = 'None',
                 group  = None,
                 uvs    = None,
                 colors = None):

        self.name     = name
        self.group    = group
        self.uvs      = uvs
        self.colors   = colors
                      
        self.nfaces   = None
        self.size     = None
        self.points   = None
        self.indicies = None

        self.setpoints( points )
        self.setindicies( indicies )

    def setpoints( self, points ):
        'force numpy array of the current float type'
        try:
            self.points = numpy.array(points).astype(DTYPE)
            if len(self.points.shape) == 1:
                self.size = len(points)/3
                self.points = self.points.reshape( (self.size, 3) )

            assert( self.points.shape[1] == 3 )
            self.size = self.points.shape[0]
        except:
            raise Warning, 'needs an array like object of float triples'

    def setindicies(self, indicies ):
        'force numpy array of the current int type'
        try:
            self.indicies = numpy.array(indicies).astype(INTDTYPE)
            if len(self.indicies.shape) == 1:
                self.nfaces = len(points)/self.verts_per_face
                self.indicies = self.indicies.reshape( (self.nfaces, self.verts_per_face) )

            assert( self.indicies.shape[1] == self.verts_per_face )
            self.nfaces = self.indicies.shape[0]
        except:
            raise Warning, 'needs an array like object of int '+self.verts_per_face_name

    def copy(self):
        'return duplicate mesh'
        cls = super(self)
        return cls(  self.points
                    ,self.indicies 
                    ,name   = self.name+'Copy'
                    ,group  = self.group
                    ,uvs    = self.uvs   
                    ,colors = self.colors )

    def __repr__(self):
        'trimesh in string form'
        return self.clsname+'('+\
            repr(self.points)[6:-1]+\
            ','+\
            repr(self.indicies)[6:-1]+\
            ', name ="'+\
            self.name+'")'

    def xformresult( self, mx ):
        'homogenious point matrix multiply, return result'
        hpoints = numpy.ones( (len(self.points), 4), dtype=DTYPE )
        hpoints[:, :3] = self.points
        hpoints = numpy.dot( hpoints, mx)
        wcoords = hpoints[:, 3]
        hpoints[:, 0] /= wcoords
        hpoints[:, 1] /= wcoords
        hpoints[:, 2] /= wcoords
        npoints = hpoints[:, :3]
        return numpy.ascontiguousarray(npoints)

    def xform( self, mx ):
        'homogenious point matrix multiply, apply result'
        self.points = self.xformresult(mx)

    def normalizepoints( self ):
        'make all points unit length -- spherify'
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        scale = 1.0 / numpy.sqrt(x*x+y*y+z*z)
        self.points[:, 0] *= scale
        self.points[:, 1] *= scale
        self.points[:, 2] *= scale
    
    def uvzeros(self):
        'init uvs'
        self.uvs = numpy.zeros( (len(self.points), 2), dtype=DTYPE )

    def color_ones(self):
        'set vertex colors to RGBA=1.0'
        self.colors = numpy.ones( (len(self.points),4), dtype=DTYPE )
