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

    def _uremap( self, u ):
        'normalize u'            
        return u / self.face_shape[0]

    def _vremap( self, v ):
        'normalize v'            
        return v / self.face_shape[1]

    def uvs( self, flip=(False,False) ): 
        'uvs -> u = axis_1, v = axis_2'
        vu = numpy.mgrid[ 0:self.vert_shape[1], 0:self.vert_shape[0]  ].astype(DTYPE)

        u = numpy.ravel( vu[1] )
        v = numpy.ravel( vu[0] )

        u = self._uremap( u )
        v = self._vremap( v )

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
    'quad mesh with sphere approximation, obj output and trimesh conversion'
    clsname = 'QuadMesh'
    verts_per_face = 4
    verts_per_face_name = 'quads'

    @classmethod
    def sphere(cls, *args, **kwargs):
        'build a grid and convertion to spherical coords'
        i = cls.grid( *args, **kwargs )
        vecutil.polar2xyz(i.points)

        return i

    @classmethod
    def grid(cls
            ,theta_faces = 31 
            ,phi_faces   = 21
            ,theta_lo    =  numpy.pi 
            ,theta_hi    = -numpy.pi 
            ,phi_lo      =  .5 * numpy.pi 
            ,phi_hi      = -.5 * numpy.pi  ):
        'construct a grid with reasonable defaults for spherical mapping'
        gd = GridDescriptor().set_face_shape( (theta_faces, phi_faces) )
        pts, idxs = gd.points_indicies()

        uvs = gd.uvs( flip=(True,False) )

        pts[:,0] = 1.0
        pts[:,1] = pts[:,1]*(theta_hi - theta_lo) + theta_lo
        pts[:,2] = pts[:,2]*(phi_hi   - phi_lo  ) + phi_lo  

        return cls( pts, idxs, uvs=uvs )

    def asTriMesh(self):
        'each quad -> 2 tris'
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
