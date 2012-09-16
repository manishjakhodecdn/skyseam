# ----------------------------------------------------------------------------
# seam
# Copyright (c) 2011 Alex Harvill
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
'image io, transformations based on numpy'

import numpy
import math

class Image(object):  
    'image class based on numpy and the netpbm formats'
    def __init__(self):
        self.data = numpy.array([],dtype=numpy.float32)

    def fromPBM( self, path ):
        'read the pbm files that photoshop writes out'
        with open(path) as fd:
            magic    = fd.readline().strip()
            width    = int( fd.readline() )
            height   = int( fd.readline() )
            max      = float( fd.readline() )
            channels = {'P5':1, 'P6':3, 'Pf':1, 'PF':3 }[magic]

            dtype    = {65535.0  : numpy.uint16
                       ,255.0    : numpy.uint8
                       ,-1.0     : numpy.float32 }[max]

           #print( magic, width, height, max, channels, dtype )

            self.data = numpy.fromfile( fd, dtype=dtype, count=width*height*channels )

            if dtype == numpy.uint16:
                self.data =  self.data.byteswap( True )

            self.data = self.data.astype( numpy.float32 )

            self.data = numpy.reshape( self.data, (channels, height, width) )

            if dtype != numpy.float32:
                self.data /= ( max - 1 )
            else:
                self.data = numpy.power( self.data, 1 / 2.2 )
                pass
           
        return self

    def smaller( self, skip = 3 ):
        print( len(self.data[0,0,:]) ) #x
        print( len(self.data[0,:,0]) ) #y
        print( len(self.data[:,0,0]) ) #chan

        self.data = self.data[:,::skip,::skip]

    def asTriMesh( self ):
        if 0 in self.data.shape:
            raise Warning('empty image')

        from trimesh import TriMesh
        grid = TriMesh.grid( self.data.shape[1] - 1, self.data.shape[2] - 1 )
        return grid

if __name__ == '__main__':
#   from trimesh import TriMesh
#   grid = TriMesh.grid( 3, 3)
#   print( grid.points.reshape((4,4,3)) )
#  #print(grid.points)
#   print(grid.points.shape)
#   print(grid.indicies.shape)
#  #import sys
#  #sys.exit()

    import os
    from glob import glob
    searchstr = os.path.dirname(__file__)+'resource/*.pfm'
    numpy.set_printoptions( precision=3, suppress=True )
    for path in glob( searchstr ):
        i = Image().fromPBM(path)
        i.smaller()
        g = i.asTriMesh()
        g.writeobj( 'test.obj' )
        print( i.data.shape, g.points.shape )
       #print( i.data[-1, :, 0], i.data.shape )
