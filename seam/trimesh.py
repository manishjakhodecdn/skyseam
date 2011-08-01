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
'triangle mesh with subdivision based on numpy'

import numpy
import math
import copy
import vecutil
from constants import DTYPE,INTDTYPE

class TriMesh(object):  
    'triangle mesh with obj output and subdivision based on numpy'
    def __init__(self, points, indicies, 
                 name = 'name', group = None, edges = None, uvs=None):
        self.name  = name
        self.group = group
        self.edges = edges
        self.uvs   = uvs
        self.v2f   = None
        self.e2f   = None
        self.f2e   = None
        self.level = 0

        self.shortedgelen = None
        self.avgedgelen   = None
        self.longedgelen  = None

        try:
            self.points = numpy.array(points).astype(DTYPE)
            if len(self.points.shape) == 1:
                self.size = len(points)/3
                self.points = self.points.reshape( (self.size, 3) )

            assert( self.points.shape[1] == 3 )
            self.size = self.points.shape[0]
        except:
            raise Warning, 'needs an array like object of float triples'

        try:
            self.indicies = numpy.array(indicies).astype(INTDTYPE)
            if len(self.indicies.shape) == 1:
                self.nfaces = len(points)/3
                self.indicies = self.indicies.reshape( (self.nfaces, 3) )

            assert( self.indicies.shape[1] == 3 )
            self.nfaces = self.indicies.shape[0]
        except:
            raise Warning, 'needs an array like object of int triples'

        self.erase    = numpy.ones( (self.nfaces), dtype = numpy.bool)

        if self.edges == None:
            self.buildedges()

    def buildedges(self):
        'table of edges to verts'
        v0 = self.indicies[:, 0]
        v1 = self.indicies[:, 1]
        v2 = self.indicies[:, 2]

        n0 = len(v0)
        n1 = n0*2
        n2 = n0*3
        edges = numpy.zeros( (n2, 2), dtype = INTDTYPE)
        edges[0 :n0, 0] = v0
        edges[0 :n0, 1] = v1
        edges[n0:n1, 0] = v1
        edges[n0:n1, 1] = v2
        edges[n1:n2, 0] = v2
        edges[n1:n2, 1] = v0

        #lowest index first
        edges = numpy.sort( edges, axis = 1  )

        #remove duplicate idx pairs
        t = edges.dtype
        view = edges.view([('', t),('', t)])
        groups = numpy.sort(view, axis = 0).view(t)
        self.edges = groups[::2]

    def buildrefedges(self):
        'table of edges to faces'
        v0 = self.indicies[:, 0]
        v1 = self.indicies[:, 1]
        v2 = self.indicies[:, 2]

        fidxs = numpy.mgrid[0:len(v0)]

        n0 = len(v0)
        n1 = n0*2
        n2 = n0*3

        edgefacedtype = numpy.dtype([ 
            ('e', [('i', INTDTYPE),('j', INTDTYPE)]), ('k', INTDTYPE)
            ])

        ef = numpy.recarray( (n2,), dtype = edgefacedtype)

        edges = numpy.zeros( (n2, 2), dtype = INTDTYPE)
        edges[0 :n0, 0] = v0
        edges[0 :n0, 1] = v1
        edges[n0:n1, 0] = v1
        edges[n0:n1, 1] = v2
        edges[n1:n2, 0] = v2
        edges[n1:n2, 1] = v0

            #lowest vert index first
        edges = numpy.sort( edges, axis = 1  )

        ef.e.i = edges[:, 0]                        
        ef.e.j = edges[:, 1]                        

        ef.k[0 :n0] = fidxs
        ef.k[n0:n1] = fidxs
        ef.k[n1:n2] = fidxs

            #group records
        efs = numpy.sort(ef)

        efs = efs.view(INTDTYPE).reshape((n2, 3))

            #each edge pair alternates at the end of the 3 tuple
            #each vert pair is repeated twice at the beginning of the 3 tuple
        f0 = efs[::2, 2]
        f1 = efs[1::2, 2]
        self.edges = efs[::2, :2]

        self.e2f = numpy.zeros( (len(f0), 2), dtype = INTDTYPE)
        self.e2f[:, 0] = f0
        self.e2f[:, 1] = f1

        faceedgedtype = numpy.dtype([ ('f', INTDTYPE), ('e', INTDTYPE) ])
        fe = numpy.recarray( (n2,), dtype = faceedgedtype)

        eidxs = numpy.mgrid[0:len(self.edges)]
        fe.f = efs[:, 2]
        fe.e[::2]  = eidxs
        fe.e[1::2] = eidxs

        fes = numpy.sort(fe)

        fes = fes.view(INTDTYPE).reshape((n2, 2))
        e0 = fes[::3, 1]
        e1 = fes[1::3, 1]
        e2 = fes[2::3, 1]

        self.f2e = numpy.zeros( (len(e0), 3), dtype = INTDTYPE)
        self.f2e[:, 0] = e0
        self.f2e[:, 1] = e1
        self.f2e[:, 2] = e2

    def buildvert2faces(self):
        'dictionary of verts to faces'
        self.v2f = {}
        for i in xrange( self.nfaces ):
            v0, v1, v2 = self.indicies[i]
            self.v2f.setdefault(v0, set([])).add(i)
            self.v2f.setdefault(v1, set([])).add(i)
            self.v2f.setdefault(v2, set([])).add(i)
    
    def edgelengths(self, points=None):
        'distance between start and end point for each edge'
        if points is None:
            points = self.points
        p0 = points[self.edges[:, 0]]
        p1 = points[self.edges[:, 1]]

        edgevecs = p0 - p1
       #return numpy.sqrt( numpy.sum( edgevecs ** 2) ) 
        x, y, z = edgevecs[:, 0], edgevecs[:, 1], edgevecs[:, 2]
        return numpy.sqrt( x*x+y*y+z*z )

    def edgestats(self):
        'return min,max,avg edge lens. compute if not already available.'

        if not all( (self.shortedgelen, self.avgedgelen, self.longedgelen) ):
            edgelengths = self.edgelengths()
            self.shortedgelen = numpy.min(edgelengths)
            self.longedgelen  = numpy.max(edgelengths)
            self.avgedgelen   = edgelengths.sum() / float(len(self.edges))

        return self.shortedgelen, self.avgedgelen, self.longedgelen 

    def antipodes( self, idx ):
        'TODO write doc str'
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
        return hpoints[:, :3]

    def xform( self, mx ):
        'homogenious point matrix multiply, apply result'
        self.points = self.xformresult(mx)

    def writeobj(self, filename ='/var/tmp/tmp.obj'):
        'write obj with uvs if available, use self.erase to delete faces'
        fd = open(filename,'w')
        numpy.savetxt( fd, self.points, delimiter =' ', fmt ='v %g %g %g')
        if self.uvs != None:
            numpy.savetxt( fd, self.uvs, delimiter =' ', fmt ='vt %g %g')

            shape = (len(self.indicies), 6)
            vertuvspec = numpy.zeros( shape, dtype = INTDTYPE )
            vertuvspec[:, 0] = self.indicies[:, 0]
            vertuvspec[:, 1] = self.indicies[:, 0]
            vertuvspec[:, 2] = self.indicies[:, 1]
            vertuvspec[:, 3] = self.indicies[:, 1]
            vertuvspec[:, 4] = self.indicies[:, 2]
            vertuvspec[:, 5] = self.indicies[:, 2]
            numpy.savetxt( fd, 
                           vertuvspec[numpy.nonzero(self.erase)] + 1,
                           delimiter =' ',
                           fmt ='f %g/%g %g/%g %g/%g')
        else:
            numpy.savetxt( fd, 
                           self.indicies[numpy.nonzero(self.erase)] + 1,
                           delimiter =' ',
                           fmt ='f %g %g %g')
        fd.close()

    def normalizepoints( self ):
        'make all points unit length -- spherize'
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        scale = 1.0 / numpy.sqrt(x*x+y*y+z*z)
        self.points[:, 0] *= scale
        self.points[:, 1] *= scale
        self.points[:, 2] *= scale

    def subdivide( self, **kwargs):
        ''' add new points and faces like so:

              p0
             /  \             1
           m0___ m2           2
          / \   / \         3   4
        p1____m1___p2       
        '''
        lo = len(self.points)
        hi = lo + len(self.edges)
        pts = numpy.zeros( (hi, 3), dtype=DTYPE )
        pts[0:lo] = self.points

        edgepts = self.points[self.edges]
        edict = {}
        for idx, edge in enumerate(self.edges):
            edict[tuple(edge)] = idx

        pts[lo:hi] = (edgepts[:, 0] + edgepts[:, 1])*.5

        nfaces = len(self.indicies)*4
        idxs = numpy.zeros( (nfaces, 3), dtype = numpy.int64)

        for oi,(p0, p1, p2) in enumerate(self.indicies):
            m0 = edict[tuple(sorted((p0, p1)))] + lo
            m1 = edict[tuple(sorted((p1, p2)))] + lo
            m2 = edict[tuple(sorted((p2, p0)))] + lo

            i = oi*4

            idxs[i+0] = (p0, m0, m2)
            idxs[i+1] = (m0, m1, m2)
            idxs[i+2] = (p1, m1, m0)
            idxs[i+3] = (m1, p2, m2)

        smesh = TriMesh(pts, idxs, **kwargs)
        smesh.level = self.level + 1
        return smesh

    def vertsinfrustum(self, w2c, w2s ):
        '''transform points to canonical camera space and clip
        return True/False numpy bool for each vert
        '''

        campoints    = self.xformresult( w2c )
        screenpoints = self.xformresult( w2s )

        x = numpy.abs(screenpoints[:, 0])
        y = numpy.abs(screenpoints[:, 1])
        z = campoints[:, 2]

        xvisible = x < 1.0 + vecutil.SMALL 
        yvisible = y < 1.0 + vecutil.SMALL 
        zvisible = z < vecutil.SMALL     

        xyvisible  = numpy.minimum( xvisible, yvisible)
        xyzvisible = numpy.minimum( xyvisible, zvisible)
    
        return xyzvisible


    def binarysignal_e2f(self, signal, combinefunc = numpy.maximum):
        '''each face gets the signal value of one edge
        use combinefunc to change the compare function for edges'''

        if self.f2e is None:
            self.buildrefedges()

        e0 = self.f2e[:, 0]
        e1 = self.f2e[:, 1] 
        e2 = self.f2e[:, 2]

        k0 = signal[e0]
        k1 = signal[e1]
        k2 = signal[e2]

        k01  = combinefunc( k0, k1  )
        return combinefunc( k2, k01 )

    def binarysignal_f2e(self, signal, combinefunc = numpy.maximum):
        '''each edge gets the signal value of one face
        use combinefunc to change the compare function for faces'''

        f0 = self.e2f[:, 0]
        f1 = self.e2f[:, 1] 

        k0 = signal[f0]
        k1 = signal[f1]

        return combinefunc( k0, k1 )

    def binarysignal_v2f(self, signal, combinefunc = numpy.maximum):
        '''each face gets the signal value of one vert
        use combinefunc to change the compare function for verts'''
        v0 = self.indicies[:, 0]
        v1 = self.indicies[:, 1] 
        v2 = self.indicies[:, 2]

        k0 = signal[v0]
        k1 = signal[v1]
        k2 = signal[v2]

        k01  = combinefunc( k0, k1  )
        return combinefunc( k2, k01 )

    def maxsignal_f2v(self, signal):
        '''each verts gets the max signal value of its faces'''
        if self.v2f == None:
            self.buildvert2faces()

        result = numpy.zeros( len(self.points), dtype=signal.dtype )
        for vert, faces in self.v2f.iteritems():
            m = numpy.NINF
            for face in faces:
                nm = signal[face]
                if nm > m:
                    m = nm
            result[vert] = m

        return result

    def avgsignal_f2v(self, signal):
        '''each verts gets the max signal value of its faces'''
        if self.v2f == None:
            self.buildvert2faces()

        result = numpy.zeros( len(self.points), dtype=signal.dtype )
        for vert, faces in self.v2f.iteritems():
            m = 0.0
            for face in faces:
                m += signal[face]
            result[vert] = m / float(len(faces))

        return result

    def copy(self):
        'return duplicate mesh'
        mesh = TriMesh(  self.points
                        ,self.indicies 
                        ,name  = self.name+'Copy'
                        ,group = self.group
                        ,edges = self.edges
                        ,uvs   = self.uvs )
        mesh.uvs   = self.uvs   
        mesh.v2f   = self.v2f   
        mesh.e2f   = self.e2f   
        mesh.f2e   = self.f2e   
        mesh.level = self.level 

        return mesh

    def polar( self, f ):
        '''transform mesh to polar coordinates
        erase the seam to eliminate coplanar faces for visualization'''
        if f is not None:
            self.xform(f.wtc)

        vecutil.xyz2polar(self.points)

        if f is not None:
            self.xform(f.ctw)
      
        edgelengths = self.edgelengths()
   
        longedges =  edgelengths < .5 * math.pi
        self.erase = self.binarysignal_e2f( longedges, numpy.minimum )
     
        return self
    
    def cells( self, level, vert=True ):
        ''' a cell is a collection of faces corresponding to
        faces of the icosa at a coarser subdiv level.
        the base icosa mesh has 20 faces
        the subivider indexes subfaces contiguously
        the each subdiv level has 4x more faces
        find the number of cells for the input level
        find the number of faces in each cell
        mark 0:subfaces = 0, subfaces:2*subfaces = 1 ...
        convert signal to verts if need be
        '''
        nfaces = len(self.indicies)
        level = max(level, 1)
        level = min(level, self.level)
    
        ncells = 20*pow(4, level-1)

        subfaces = nfaces/(ncells)
        fsig = numpy.zeros( nfaces, dtype=numpy.int32 )

        for i in range(ncells):
            fsig[i * subfaces:(i + 1) * subfaces] = i

        if vert:
            return self.maxsignal_f2v( fsig ), ncells
            
        return fsig, ncells

    def uvzeros(self):
        'init uvs'
        self.uvs = numpy.zeros( (len(self.points), 2), dtype=DTYPE )

    def __repr__(self):
        'trimesh in string form'
        return 'TriMesh('+\
            repr(self.points)[6:-1]+\
            ','+\
            repr(self.indicies)[6:-1]+\
            ', name ="'+\
            self.name+'")'

    @staticmethod
    def cube():
        'a triangulated unit cube'
           
        verts = numpy.zeros( (8, 3), dtype=DTYPE )
        c = 0
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    verts[c] = (i, j, k)
                    c += 1

        indicies = numpy.array( [[0, 1, 2], [1, 3, 2]
                                ,[0, 4, 5], [0, 5, 1]
                                ,[4, 6, 7], [4, 7, 5]
                                ,[6, 2, 3], [6, 3, 7]
                                ,[0, 6, 4], [0, 2, 6]
                                ,[1, 7, 3], [1, 5, 7]], dtype=INTDTYPE )

        return TriMesh( verts, indicies)

    @staticmethod
    def icosa():
        '''  12 verts, 20 faces, regular topology
        (0, +-1, +-p)
        (+-1, +-p, 0)
        (+-p, 0, +-1) 
        '''
        p = (1.0 + math.sqrt(5))*.5

        pts = numpy.zeros((12, 3), dtype=DTYPE)
        c = 0
        i = 0.
        for j in (-1, 1):
            for k in (-p, p):
                pts[c+0] = vecutil.vec3(i, j, k)             
                pts[c+1] = vecutil.vec3(j, k, i)             
                pts[c+2] = vecutil.vec3(k, i, j)             
                c += 3

        pts *= 1.0 / vecutil.vlength(pts[0])

            #top:3, bottom:6
        mesh = numpy.array(( ( 0, 1,  2), ( 0, 2,  6)
                            ,( 8, 2,  1), ( 8, 1,  3)
                            ,( 2, 8,  4), ( 2, 4,  6)
                            ,( 9, 4,  8), ( 9, 8,  3)
                            ,( 4, 9, 10), ( 4, 10, 6)
                            ,(11, 10, 9), (11, 9,  3)
                            ,(10, 11, 5), (10, 5,  6)
                            ,( 7, 5, 11), ( 7, 11, 3)
                            ,( 5, 7,  0), ( 5, 0,  6)
                            ,( 1, 0,  7), ( 1, 7,  3)
                         ), dtype=INTDTYPE)

        return TriMesh( pts, mesh)

    @staticmethod
    def grid(tfaces=40, pfaces=20):
        '''tri grid for spherical coordinates
        ntris = tfaces * pfaces * 2
        radius:          x == 1
        theta : -pi   <= y <= pi
        phi   : -pi/2 <= z <= pi/2
        '''
        tverts = tfaces + 1
        pverts = pfaces + 1

        basis = numpy.mgrid[ :pfaces, :tfaces, :3]

        y = numpy.ravel( basis[0] )
        x = numpy.ravel( basis[1] )
        n = numpy.ravel( basis[2] )

        i0   = n == 0
        i1   = n == 1
        i2   = n == 2

        a = y*tverts+x + i1 + tverts * i2
        b = y*tverts+x + i0 + tverts * i1 + tverts * i2 + i2
    
        idxs = numpy.cast[INTDTYPE](numpy.concatenate( (a, b[::-1]) ) )
        idxs = idxs.reshape( (len(idxs)/3, 3) )

        vu = numpy.cast[DTYPE]( numpy.mgrid[ 0:pverts, 0:tverts  ] )

        p = numpy.ones( (tverts*pverts, 3), dtype=DTYPE)
        p[:,1] = numpy.ravel( vu[1] )
        p[:,2] = numpy.ravel( vu[0] )

        p[:,1] /= tfaces
        p[:,2] /= pfaces
        p[:,1] -= .5
        p[:,2] -= .5

        p[:,1] *= 2 * numpy.pi
        p[:,2] *= numpy.pi

        return TriMesh( p, idxs )

if __name__ == '__main__':
    print( TriMesh.cube() )
    print( TriMesh.icosa() )
