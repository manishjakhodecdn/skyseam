# ----------------------------------------------------------------------------
# seam
# Copyright (c) 2011-2012 Alex Harvill
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions 
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright 
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of seam nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------
'common parts of TriMesh and QuadMesh'
import numpy
from constants import DTYPE,INTDTYPE

class FixedMesh(object):  
    'base mesh class'
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

        self.erase    = None
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
                self.nfaces = len(self.indicies)/self.verts_per_face
                self.indicies = self.indicies.reshape( (self.nfaces, self.verts_per_face) )

            assert( self.indicies.shape[1] == self.verts_per_face )
            self.nfaces = self.indicies.shape[0]
            self.erase = numpy.ones( (self.nfaces), dtype = numpy.bool)
        except:
            raise Warning, 'needs an array like object of int '+self.verts_per_face_name

    def copy(self):
        'return duplicate mesh'
        cls = super(FixedMesh, self)
        return cls(  self.points
                    ,self.indicies 
                    ,name   = self.name+'Copy'
                    ,group  = self.group
                    ,uvs    = self.uvs   
                    ,colors = self.colors )

    def __repr__(self):
        'trimesh in string form'
        startstrs = ('array(',)
        endstrs   = (', dtype=int32)',', dtype=int64)',', dtype=float32)',', dtype=float64)')
        def trimstr(pts, startstrs, endstrs):
            for startstr in startstrs:
                if pts.startswith(startstr):
                    pts = pts[len(startstr):]
            for endstr in endstrs:
                if pts.endswith(endstr):
                    pts = pts[:-1*len(endstr)]
            return pts

        pts  = trimstr( repr(self.points), startstrs, endstrs)
        idxs = trimstr( repr(self.indicies), startstrs, endstrs)

        return ''.join(( self.clsname, '(', pts, ',\n      ', idxs, ',name="', self.name, '")' ))

    def antipodes( self, idx ):
        'finds the point reflected about the origin'
        inv = self.points[idx] * -1
        dist = numpy.abs( numpy.inner( inv, self.points) - 1 )
        return numpy.argmin( dist )

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

    def writeobj(self, filename ='/var/tmp/tmp.obj'):
        'write obj with uvs if available, use self.erase to delete faces'
        fd = open(filename, 'w')
        numpy.savetxt( fd, self.points, delimiter =' ', fmt ='v %g %g %g')
        if self.uvs != None:
            numpy.savetxt( fd, self.uvs, delimiter =' ', fmt ='vt %g %g')

            #repeat vert indicies 2x (1st verts then uvs)
            shape = (len(self.indicies), self.verts_per_face*2)
            vertuvspec = numpy.zeros( shape, dtype = INTDTYPE )

            for i in range(self.verts_per_face):
                j=i*2
                vertuvspec[:, j+0] = self.indicies[:, i]
                vertuvspec[:, j+1] = self.indicies[:, i]

            facefmt = ['f'] + ['%g/%g']*self.verts_per_face
            numpy.savetxt( fd, 
                           vertuvspec[numpy.nonzero(self.erase)] + 1,
                           delimiter =' ',
                           fmt =' '.join(facefmt))
        else:
            facefmt = ['f'] + ['%g']*self.verts_per_face
            numpy.savetxt( fd, 
                           self.indicies[numpy.nonzero(self.erase)] + 1,
                           delimiter =' ',
                           fmt =' '.join(facefmt))
        fd.close()
