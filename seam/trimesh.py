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
import vecutil
from constants import DTYPE, INTDTYPE

class TriMesh(object):  
    'triangle mesh with obj output and subdivision based on numpy'
    def __init__(self, points, indicies, 
                 name = 'name', group = None, edges = None, uvs=None, f2e=None):
        self.name  = name
        self.group = group
        self.edges = edges
        self.uvs   = uvs
        self.v2f   = None
        self.e2f   = None
        self.f2e   = f2e
        self.level = 0

        self.closed = None

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

        self.erase = numpy.ones( (self.nfaces), dtype = numpy.bool)

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
        view = edges.view([('', t), ('', t)])
        groups = numpy.sort(view, axis = 0).view(t)
        self.edges = groups[::2]

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
        'return min, max, avg edge lens. compute if not already available.'

        if not all( (self.shortedgelen, self.avgedgelen, self.longedgelen) ):
            edgelengths = self.edgelengths()
            self.shortedgelen = numpy.min(edgelengths)
            self.longedgelen  = numpy.max(edgelengths)
            self.avgedgelen   = edgelengths.sum() / float(len(self.edges))

        return self.shortedgelen, self.avgedgelen, self.longedgelen 

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
        return hpoints[:, :3]

    def xform( self, mx ):
        'homogenious point matrix multiply, apply result'
        self.points = self.xformresult(mx)

    def writeobj(self, filename ='/var/tmp/tmp.obj'):
        'write obj with uvs if available, use self.erase to delete faces'
        fd = open(filename, 'w')
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

    def edgehash( self, edges ):
        'ravel vert idxs together to create a unique int per edge'
        maxidx = len( self.points )
        edges = numpy.sort( edges, axis=1)
        edgehash = edges[:, 0] * maxidx + edges[:, 1]
        return edgehash

    def f2eslow( self ):
        '''straightforward method to build the face to edge table
        a dictionary maps 2 points that form an edge to an edge index
        for each face, lookup edges from points
        this is slow because a ton of logic is happening per face/edge'''
        p0 = self.indicies[:, 0]
        p1 = self.indicies[:, 1]
        p2 = self.indicies[:, 2]

        edict = {}
        for idx, edge in enumerate(self.edges):
            edict[tuple(edge)] = idx

        self.f2e = self.indicies.copy()
        for oi, (p0, p1, p2) in enumerate(self.indicies):
            self.f2e[oi, 0] = edict[tuple(sorted((p0, p1)))]
            self.f2e[oi, 1] = edict[tuple(sorted((p1, p2)))]
            self.f2e[oi, 2] = edict[tuple(sorted((p2, p0)))]

    def f2efast( self ):
        '''optimized version of f2eslow
        an intermediate hash number is calculated for each edge
        logic now happens outside the python loops'''

        try:
            hashededges = self.edgehash( self.edges )

            edgelookup = {}
            for idx, hashededge in enumerate(hashededges):
                edgelookup[hashededge] = idx
       
            p0 = self.indicies[:, 0 ]
            p1 = self.indicies[:, 1 ]
            p2 = self.indicies[:, 2 ]
            e0 = numpy.vstack( (p0, p1) ).transpose()
            e1 = numpy.vstack( (p1, p2) ).transpose()
            e2 = numpy.vstack( (p2, p0) ).transpose()

            e0 = self.edgehash(e0)
            e1 = self.edgehash(e1)
            e2 = self.edgehash(e2)
    
            f2e = numpy.zeros( self.indicies.shape, dtype=INTDTYPE )
            for fidx in xrange( len(e0) ):
                f2e[fidx, 0] = edgelookup[ e0[fidx] ]
                f2e[fidx, 1] = edgelookup[ e1[fidx] ]
                f2e[fidx, 2] = edgelookup[ e2[fidx] ]

            self.f2e = f2e
            self.closed = True
        except KeyError:
            self.closed = False

    def e2ffast(self):
        'compute the edge2faces table from the face2edge table'
        if self.f2e is None:
            self.f2efast()
            if self.closed == False:
                return

        idxs = numpy.mgrid[0:len(self.indicies)].astype( INTDTYPE )
        alledges = self.f2e.transpose().ravel()
        allfaces = numpy.concatenate( (idxs, idxs, idxs) )

        e2f = numpy.argsort( alledges )

        e2f = allfaces[e2f] #otherwise idxs run from 0 - nfaces * 3
 
        self.e2f = e2f.reshape( self.edges.shape )

    def subdivide( self, **kwargs):
        ''' add new points and faces like so:

              p0
             /  \             1
           m0___ m2           2
          / \   / \         3   4
        p1____m1___p2       
        '''


        if self.f2e == None:
            self.f2efast()
    
        if self.closed == False:
            raise Warning('cannot subdivide an open mesh')
            

        lo = len(self.points)
        hi = lo + len(self.edges)
        pts = numpy.zeros( (hi, 3), dtype=DTYPE )
        pts[0:lo] = self.points

        edgepts = self.points[self.edges]
        pts[lo:hi] = (edgepts[:, 0] + edgepts[:, 1])*.5

        nfaces = len(self.indicies)*4
        idxs = numpy.zeros( (nfaces, 3), dtype = INTDTYPE )

        p0 = self.indicies[:, 0]
        p1 = self.indicies[:, 1]
        p2 = self.indicies[:, 2]

        m0 = self.f2e[:, 0] + lo
        m1 = self.f2e[:, 1] + lo
        m2 = self.f2e[:, 2] + lo

        idxs[0::4, 0] = p0
        idxs[0::4, 1] = m0
        idxs[0::4, 2] = m2
        idxs[1::4, 0] = m0
        idxs[1::4, 1] = m1
        idxs[1::4, 2] = m2
        idxs[2::4, 0] = p1
        idxs[2::4, 1] = m1
        idxs[2::4, 2] = m0
        idxs[3::4, 0] = m1
        idxs[3::4, 1] = p2
        idxs[3::4, 2] = m2

            # 2 exterior edges per orig edge + 3 interior edges per face
        outeredges = len(self.edges) * 2
        nedges = len(self.edges) * 2 + len(self.indicies) * 3
        edges = numpy.zeros( (nedges, 2), dtype = INTDTYPE )
        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]
        en = numpy.mgrid[0:len(self.edges)] + lo
        edges[0:outeredges:2, 0] = e0
        edges[0:outeredges:2, 1] = en
        edges[1:outeredges:2, 0] = en
        edges[1:outeredges:2, 1] = e1

        edges[outeredges::3, 0] = m0
        edges[outeredges::3, 1] = m1
        edges[outeredges+1::3, 0] = m1
        edges[outeredges+1::3, 1] = m2
        edges[outeredges+2::3, 0] = m2
        edges[outeredges+2::3, 1] = m0

        edgeflip = edges[:, 0] > edges[:, 1]
   
        tmp = edges[ edgeflip ]
        flip = tmp.copy()
        tmp[:, 0] = flip[:, 1]
        tmp[:, 1] = flip[:, 0]
        edges[ edgeflip ] = tmp

        smesh = TriMesh(pts, idxs, edges=edges, **kwargs)

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
            self.f2efast()

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

        if self.e2f is None:
            self.e2ffast()

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

        c = TriMesh( verts, indicies )

        return c

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

        imesh = TriMesh( pts, mesh )
        return imesh

    @staticmethod
    def grid(tfaces=40, pfaces=20, int_coords=False):
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
        p[:, 1] = numpy.ravel( vu[1] )
        p[:, 2] = numpy.ravel( vu[0] )
 
        if int_coords:
            p[:, 1] /= tfaces
            p[:, 2] /= pfaces
            p[:, 1] -= .5
            p[:, 2] -= .5

            p[:, 1] *= 2 * numpy.pi
            p[:, 2] *= numpy.pi

        return TriMesh( p, idxs )

#if __name__ == '__main__':
#
#   def navgsignal_f2v( self, frand, sv2f, v2sv ):
#       nvrand = numpy.zeros( len(self.points), dtype=DTYPE )
#       
#       for i,sub in enumerate(sv2f):
#           n = v2sv[i]
#           print(sub.shape)
#           a = numpy.zeros( sub.shape[0], dtype=DTYPE)
#           for j in range(sub.shape[-1]):
#               a += frand[sub[:,j]]
#           a  *= (1.0 / float(sub.shape[-1]) )
#           nvrand[n] = a
#       return nvrand

#   def testsubd():
#       c = TriMesh.icosa()
#      #c = TriMesh.cube()
#      #c.buildrefedges()
#       c = c.subdivide()
#       c = c.subdivide()
#       c = c.subdivide()
#       c = c.subdivide()
#       c = c.subdivide()
#       c = c.subdivide()
#      #c = c.subdivide()
#       for i in range(20):
#           c.binarysignal_f2e( numpy.mgrid[0:len(c.indicies) ])
#       print(c.points.shape)
#      #c = c.subdivide()
#      #print(c)

#   def test():
#   #    print( TriMesh.cube() )
#   #    print( TriMesh.icosa() )
#      #c = TriMesh.cube()
#       c = TriMesh.icosa()

#       for i in range(5):
#           c = c.subdivide()
#      #c = TriMesh.cube()
#       c.buildvert2faces()
#       maxlen = 0
#       minlen = float('inf')

#       lens = numpy.ones( (len(c.points)), dtype = INTDTYPE )
#       for v,s in c.v2f.iteritems():
#           lens[v] = len(s)

#       maxlen = lens.max()
#       minlen = lens.min()

#       sv2f = []
#       v2sv = []

#       for i in range( minlen, maxlen+1):
#           n = numpy.nonzero( lens == i )[0]
#           v2sv.append( n )
#           v2f = numpy.ones( (len(n), i), dtype = INTDTYPE )
#           for j in xrange( len( n ) ):
#                v2f[j] = tuple(c.v2f[n[j]])

#           sv2f.append( v2f )

#       frand = numpy.random.uniform( 0, 1, len(c.indicies) )

#       vrand = c.avgsignal_f2v(frand)

#       nvrand = navgsignal_f2v( c, frand, sv2f, v2sv )

#       print(numpy.sum( nvrand - vrand) )
#   import cProfile
#   cProfile.run('testsubd()')
#  #testsubd()
#   def testedges():
#       c = TriMesh.icosa()
#       c.binarysignal_f2e( numpy.mgrid[0:len(c.indicies) ])
#  #testsubd()
