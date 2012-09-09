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
'limited quad mesh based on numpy'

import numpy
import vecutil
from constants import DTYPE, INTDTYPE
from fixedmesh import FixedMesh

class GridDescriptor( object ):
    def __init__(self ):
        'describe number of faces / verts in a mesh'
        self.vert_shape = numpy.array( (0, 0), dtype=INTDTYPE  )
        self.face_shape = numpy.array( (0, 0), dtype=INTDTYPE  )
        self.n_faces    = 0
        self.n_verts    = 0
    
    def set_vert_shape(self, value):
        self.vert_shape = numpy.array( value, dtype=INTDTYPE  )
        self.face_shape = self.vert_shape - 1

        assert( self.face_shape.shape == (2,) )
        assert( (self.vert_shape > 1).all() )

        self.n_faces    = numpy.product( self.face_shape )
        self.n_verts    = numpy.product( self.vert_shape )
        return self

    def set_face_shape(self, value):
        return self.set_vert_shape( numpy.array( value, dtype=INTDTYPE  ) + 1 )

    def indicies( self ):
        'face -> vertex mapping'
        fidxs = numpy.zeros( (self.n_faces, 4 ), dtype=INTDTYPE  )

        vidxs = numpy.arange( self.n_verts, dtype=INTDTYPE ).reshape( self.vert_shape[::-1] )

        fidxs[:,0] = vidxs[  :-1,  :-1].ravel()
        fidxs[:,1] = vidxs[  :-1, 1:  ].ravel()
        fidxs[:,2] = vidxs[ 1:  , 1:  ].ravel()
        fidxs[:,3] = vidxs[ 1:  ,  :-1].ravel()
        return fidxs

    def uvs( self, bspline=False, flip=(False,False) ): 
        '''uvs -> u = axis_1, v = axis_2
        for bspline patch - coords a unit square for all coords but the edges'''
        vu = numpy.mgrid[ 0:self.vert_shape[1], 0:self.vert_shape[0]  ].astype(DTYPE)

        u = numpy.ravel( vu[1] )
        v = numpy.ravel( vu[0] )

        if bspline:
            u = ( u - 1 ) / ( self.face_shape[0] - 2 )
            v = ( v - 1 ) / ( self.face_shape[1] - 2 )
        else:
            u = u / self.face_shape[0]
            v = v / self.face_shape[1]

        if flip[0]:
            u = u * -1 + 1

        if flip[1]:
            v = v * -1 + 1

        p = numpy.zeros( (self.n_verts, 2), dtype=DTYPE)
        p[:,0] = u
        p[:,1] = v

        return p

    def points( self ): 
        'points -> x = 0, y=axis_1 z=axis_2'
        yz = numpy.mgrid[ 0:self.vert_shape[1], 0:self.vert_shape[0]  ].astype(DTYPE)

        p = numpy.zeros( (self.n_verts, 3), dtype=DTYPE)
        p[:,1] = numpy.ravel( yz[1] ) / self.face_shape[0]
        p[:,2] = numpy.ravel( yz[0] ) / self.face_shape[1]

        return p

    def points_indicies( self ):
        'returns a tuple of (points, indicies)'
        return self.points(), self.indicies()

class QuadMesh(FixedMesh):  
    'quad mesh with obj output'
    clsname = 'QuadMesh'
    verts_per_face = 4
    verts_per_face_name = 'quads'

    @classmethod
    def sphere(cls 
              ,theta_faces = 33 
              ,phi_faces   = 23
              ,theta_lo    =  numpy.pi 
              ,theta_hi    = -numpy.pi 
              ,phi_lo      =  .5 * numpy.pi 
              ,phi_hi      = -.5 * numpy.pi  ):


        gd     = GridDescriptor().set_face_shape( (theta_faces, phi_faces) )
        idxs   = gd.indicies()
        uvs    = gd.uvs( bspline=True, flip=(True,False))

        pt     = numpy.array([[1., 0., 0., 1.]], dtype=numpy.float64)

        xy_arc = vecutil.revolve_bspline_fast( points    = pt
                                              ,n_verts_u = gd.vert_shape[1]
                                              ,n_verts_v = 1
                                              ,start     = phi_lo
                                              ,end       = phi_hi)
            
        yz_arc = vecutil.rotate(xy_arc, numpy.radians(90.0), 'x')

        points = vecutil.revolve_bspline_fast( points    = yz_arc
                                              ,n_verts_u = gd.vert_shape[0]
                                              ,n_verts_v = gd.vert_shape[1]
                                              ,start     = theta_lo
                                              ,end       = theta_hi)

        points = points[:,:,:3].reshape( (gd.n_verts, 3) )

        return cls( points.ravel(), idxs, uvs=uvs )

    @classmethod
    def grid(cls, tfaces=40, pfaces=20, xwidth=2.0*numpy.pi, ywidth=numpy.pi, radius=1.0, axis=(0,1,2) ):
        gd = GridDescriptor().set_face_shape( (tfaces, pfaces) )
        pts, idxs = gd.grid_data()

        uvs = pts[:,1:].copy()

        pts[:,0] = radius
        pts[:,1] = (pts[:,1] - .5) * xwidth
        pts[:,2] = (pts[:,2] - .5) * ywidth

        return cls( pts, idxs, uvs=uvs ).swap_axes( axis )

    def asTriMesh(self):
        import trimesh
        
        idxs = numpy.zeros((self.nfaces * 2, 3), dtype=INTDTYPE)
        idxs[::2,0] = self.indicies[:,0]
        idxs[::2,1] = self.indicies[:,1]
        idxs[::2,2] = self.indicies[:,2]

        idxs[1::2,0] = self.indicies[:,2]
        idxs[1::2,1] = self.indicies[:,3]
        idxs[1::2,2] = self.indicies[:,0]

        return trimesh.TriMesh( self.points,
                                idxs,
                                name=self.name+'Tri',
                                group=self.group,
                                uvs=self.uvs)
