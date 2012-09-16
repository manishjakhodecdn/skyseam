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
'limited bspline patch based on numpy'

import numpy
import vecutil
from constants import DTYPE, INTDTYPE
from quadmesh import GridDescriptor, QuadMesh

class PatchDescriptor( GridDescriptor ):
    def _uremap( self, u ):
        'normalize u - edges are outsize unit square'            
        return ( u - 1 ) / ( self.face_shape[0] - 2 )

    def _vremap( self, v ):
        'normalize v - edges are outsize unit square'            
        return ( v - 1 ) / ( self.face_shape[1] - 2 )

class BSplinePatch(QuadMesh):
    'bspline patch with accurate spherical approximation and obj output'
    clsname = 'BSplinePatch'

    @classmethod
    def sphere(cls 
              ,theta_faces = 33 
              ,phi_faces   = 23
              ,theta_lo    =  numpy.pi 
              ,theta_hi    = -numpy.pi 
              ,phi_lo      =  .5 * numpy.pi 
              ,phi_hi      = -.5 * numpy.pi  ):


        pd     = PatchDescriptor().set_face_shape( (theta_faces, phi_faces) )
        idxs   = pd.indicies()
        uvs    = pd.uvs(flip=(True,False))

        pt     = numpy.array([[1., 0., 0., 1.]], dtype=numpy.float64)

        xy_arc = vecutil.revolve_bspline_fast( points    = pt
                                              ,n_verts_u = pd.vert_shape[1]
                                              ,n_verts_v = 1
                                              ,start     = phi_lo
                                              ,end       = phi_hi)
            
        yz_arc = vecutil.rotate(xy_arc, numpy.radians(90.0), 'x')

        points = vecutil.revolve_bspline_fast( points    = yz_arc
                                              ,n_verts_u = pd.vert_shape[0]
                                              ,n_verts_v = pd.vert_shape[1]
                                              ,start     = theta_lo
                                              ,end       = theta_hi)

        points = points[:,:,:3].reshape( (pd.n_verts, 3) )

        return cls( points.ravel(), idxs, uvs=uvs )
