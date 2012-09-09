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
'test quad mesh'

def test_sphere():
    import quadmesh
    import numpy
    i = quadmesh.QuadMesh.sphere( theta_lo = -.5 * numpy.pi
                                 ,theta_hi =  .5 * numpy.pi 
                                 ,phi_lo   = -.25 * numpy.pi  
                                 ,phi_hi   =  .25 * numpy.pi )
    i.writeobj()

def test_obj():
    import quadmesh
    i = quadmesh.QuadMesh.square()#axis=(1,2,0))
    i.writeobj()

def test_asTriMesh():
    import quadmesh
    i = quadmesh.QuadMesh.grid()#axis=(1,2,0))
    print(i)
   #t = i.asTriMesh()
   #t.writeobj()

def test_grid():
    import vecutil
    import quadmesh
    o = vecutil.meshtest()

    pts, idx = quadmesh.GridDescriptor().set_face_shape( (22, 12) ).grid_data()
    print( (idx == o).all() )

if __name__ == '__main__':
#   test_asTriMesh()

    test_sphere()
