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

class QuadMesh(FixedMesh):  
    'quad mesh with obj output'
    clsname = 'QuadMesh'
    verts_per_face = 4
    verts_per_face_name = 'quads'

    @classmethod
    def square(cls, axis=(0,1,2)):
        'single quad face'
        p = numpy.array( ((0,0,0)
                         ,(1,0,0)
                         ,(1,1,0)
                         ,(0,1,0)), dtype=DTYPE)

        uvs = p[:,:2]

        np = p.copy()
        np[:,0] = p[:,axis[0]]
        np[:,1] = p[:,axis[1]]
        np[:,2] = p[:,axis[2]]

        idxs = numpy.array(( 0, 1, 2, 3), dtype=INTDTYPE)

        return cls( np, idxs, uvs=uvs )
