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
'''perform view unwrapping operations using a subdivided icosahedron'''

import numpy
import math
import vecutil as num
from frustum import Frustum
from trimesh import TriMesh
from constants import DTYPE,INTDTYPE

class SeamSolver( object ):
    'attempt to find seams that best unwrap a sphere for specified views'
    kzup   = 'kzup'
    kplane = 'kplane'
    kfree  = 'kfree'
    kinds  = (kzup, kplane, kfree)

        # search frustum constants
    slicetolerance_halfwidth = .75  # width: .75 * maxedgelen * 2
    slicetolerance_end       = .125 # backup around pole: .125 * maxedgelen

    def __init__(self, levels = 5, kind = kzup):
        if kind not in self.kinds:
            raise Warning('%s is not one of %s '%(kind, ','.join(self.kinds)))

        self.levels       = levels
        self.freelevels   = 1
        self.kind         = kind
        self.mesh         = None
        self.iters        = None
        self.vertvis      = None
        self.vertweight   = None
        self.distantverts = None
        self.orientmx     = None
        self.vertunwrap   = None
        self.lenunwrap    = None

        self.buildmesh(levels)

    def run(self):
        ''' example usage:
        solver = SeamSolver( levels, SeamSolver.kzup )
        solver.markfrustums(frustums)
        seams = solver.run()

        returns a list of tuples describing valid seams:
            score, visibileverts, frustum
           the result is sorted by score and the lowest score is the best seam
        '''
        dispatcher = { self.kzup   : self.zupsearch, 
                       self.kplane : self.planesearch, 
                       self.kfree  : self.freesearch }

        return dispatcher[self.kind]()

    def zupsearch( self ):
        'polesearch uses the z axis when orientmx is the identity '
        self.orientmx = numpy.identity(4, dtype=DTYPE)
        return self.polesearch()

    def planesearch( self ):
        '''
        mirrors all visible verts to build a vector array symetrical about origin
        uses svd to fit a plane to the visible verts
        saves the plane rotation in orientmx
        calls polesearch to find the best seam

        info on svd for plane fitting:
        from http://en.wikipedia.org/wiki/Singular_value_decomposition
        Total least squares minimization
        A total least squares problem refers to determining the vector x which 
        minimizes the 2-norm of a vector Ax under the constraint ||x|| = 1.
        The solution turns out to be the right singular vector of A
        corresponding to the smallest singular value.
        
        from doc for numpy.linalg.svd:
        in numpy the singular values are sorted so that s[i] >= s[i+1].
        ''' 
        vispts = self.mesh.points[ self.vertvis ]

        m = numpy.concatenate( (vispts, -1*vispts) )
       
        self.orientmx = numpy.identity(4, dtype=DTYPE)
        self.orientmx[:3, :3] = numpy.linalg.svd(m)[-1]
        
        return self.polesearch()

    def freesearch(self):
        ''' free form seam finding routine
        find the vertexes with the highest weight
        for eaach the high weight verts, do a 360 degree search for best seam
        return result of those searches
        '''

        self.distantverts = self.gridmin( self.vertweight, self.freelevels )

        seams = []
        for distantvert in self.distantverts:

            self.orientmx = num.vecspace( self.mesh.points[distantvert] )
            self.vertunwrap, self.lenunwrap = \
                num.unwrap1point(self.mesh, self.orientmx, distantvert)

            seams.extend( self.ringsearch() )

        seams.sort()
   
        return seams
    
    def gridmin( self, vertsignal, level ):
        '''find the min value of input vertsignal for each cell.
        for each gridcell find the index of min value of vertsignal
        '''
        vertsbyface, ncells = self.mesh.cells( level )

        minverts = []
        for i in range(ncells):
            faceverts = vertsbyface == i
            faceverts = numpy.nonzero(faceverts)[-1] # correct range

            faceweights = vertsignal[faceverts]
            facemin = faceverts[numpy.argmin( faceweights )]
            if not self.vertvis[facemin]:
                minverts.append( facemin )

        return minverts

    def polesearch(self):
        ''' finds best seam aligned to the +-z poles of self.orientmx
        get an array of verts for each possible seam
        if the seam intersects a view, discard
        remaining seams are scored by summing all weights normalized by num samples
        the lowest score is the best seam
        '''

        result = []

        f, dummy, segs = self.buildsearchfrustums()

        for i in range(segs*2):

            t = (i / float(segs))
            vis  = self.vertsinslice( t, f )

            if not numpy.any( self.vertvis[vis] ):
                score = numpy.sum( self.vertweight[vis] )
                score /= len(vis)
                result.append( (score, vis, f.copy()) )

        result.sort()
        return result

    def vertsinslice( self, t, f ):
        'hit test mesh with a frustum rotated by t steps'
        nf = f.copy()

        r      = num.zrotatematrix( t*math.pi)
        tr     = numpy.dot( num.translatematrix( 0, 0, 2 ), r )
        nf.ctw = numpy.dot( tr, self.orientmx )
        
        sr     = num.zrotatematrix( t*math.pi )
        f.ctw  = numpy.dot( sr, self.orientmx )

        visibleverts = self.mesh.vertsinfrustum( nf.wtc, nf.wts)
        visibletoall = numpy.nonzero( visibleverts )[0]
        return visibletoall

    def ringsearch(self):
        '''find best seam for a mesh around input pivot vec
        the width of each ring is 1.35 times longest input mesh edge:edgestep
        call find closestedge to limit the ring span
        use findbestseam to position the seam in the limited span'''
        result = []

        fl, fr, segs = self.buildsearchfrustums()

        for i in range(segs):

            visible = []

            for f in (fl, fr): 
                edgerange = self.findclosestedge( i / float(segs), f )
                visible.append( edgerange )

            if visible[0] is None or visible[1] is None:
                continue

            seam = self.findbestseam( visible )
            if seam is not None:

                score, dummy, pivot, hi, vis = seam

                pole = self.mesh.points[hi]
                mid  = self.mesh.points[pivot]

                f = Frustum()
                f.ctw = num.vecspace2( pole, mid )
                result.append( (score, vis, f) )

        return result

    def findclosestedge( self, t, f ):
        ''' get list of points in current slice of sphere (w2c) 
        sort points based on arc distance
        get visiblity into for sorted points: viewedgesort
        the first derivative of viewedgesort is nonzero at edges: zerocrossings
        the min index of zerocrossings is the closest edge
        convert min index to mesh indicies
        '''

        visibletoall = self.vertsinslice( t, f )

        lengthsvis = self.lenunwrap[visibletoall]

        sortargs = numpy.argsort( lengthsvis )
        visiblesort = visibletoall[sortargs]

        viewedgesort = self.vertvis[visiblesort]
        
        delta = viewedgesort[1:] - viewedgesort[:-1]

        zerocrossings = numpy.nonzero(delta)[0]
        if len(zerocrossings) == 0:
            return None

        closestedge = numpy.min(zerocrossings)
        
        inrange = visibletoall[sortargs[:closestedge]]

        return inrange

    def findbestseam( self, visible ):
        '''visible contains 2 sets of vert indexes
        reverse the first set and concat with the last set to get a whole ring
        call minspansearch to find the best seam amongst a > 180 degree span
        if the min/max of the best seam are not antipodal, pick best antipodes
        '''
        v0 = visible[0]
        v1 = visible[1]
        
        v2 = numpy.concatenate( (v0[::-1], v1) )

        seamlength = self.lenunwrap[v2[0]] + self.lenunwrap[v2[-1]]
        if seamlength < math.pi:
            return None

        x = self.lenunwrap[v2]
        x[0:len(v0)] *= -1
        y = self.vertweight[v2]

        spanweight, lo, pivot, hi = num.minspansearch( x, y )

        lo, pivot, hi = v2[lo], v2[pivot], v2[hi]

        ahi = self.mesh.antipodes( lo )
        alo = self.mesh.antipodes( hi )

        if alo != lo:
            c0, c1, c2, c3 = self.vertvis[ [alo, hi, ahi, lo] ]

            w0, w1, w2, w3 = self.vertweight[ [alo, hi, ahi, lo] ]
            if (c0 or c1) and (c2 or c3):
                print('warning: seam crosses view')
                return None
                
            if w0 + w1 < w2 + w3:
                lo = alo
            else:
                hi = ahi

        return spanweight, lo, pivot, hi, v2

    def markfrustums( self, frustums ):
        '''finds distance along mesh per vert to nearest frustum:
        clip verts by the union of all input frustums [vertvis]
        push vertvis to faces [fsig]
        using face->edge and edge->face, grow fsig 1 step [fsign]
        if fsig and fsign are different
            add them together [ftotal]
            keep going
        
        ftotal has the range 0 - # of iterations so normalize [remap]

        push remap to verts - not numpy friendly - slow op
        '''

        mesh = self.mesh
        vertvis = numpy.zeros( (mesh.size), dtype = numpy.bool)

        for frustum in frustums:
            tmp = mesh.vertsinfrustum( frustum.wtc, frustum.wts )
            vertvis[tmp] = True

        mesh.buildrefedges()

        fsig = mesh.binarysignal_v2f(vertvis)

        ftotal = numpy.zeros(fsig.shape, dtype=DTYPE)

        i = 0
        for i in xrange(1000000):
            esig  = mesh.binarysignal_f2e(fsig)
            fsign = mesh.binarysignal_e2f(esig)
            if numpy.all(fsig == fsign):
                break
            ftotal += fsig
            fsig = fsign


        scl = (1.0/float(i))
        remap = 1.0 - ( ftotal * scl )

        vertweight = mesh.avgsignal_f2v( remap)

        self.iters      = i
        self.vertvis    = vertvis
        self.vertweight = 1-vertweight

    def buildsearchfrustums( self ):
        '''
        left and right hand ortho views that approximate a half circle of verts
        input mesh edges are not aligned to view so add some tolerance:
            top/bottom edges make a frustrum 1.5 times bigger wrt longest edge 
            outer edge is beyond the edge of the sphere
            inner edge is just slightly overlapping to capture the center point
        '''
        hi = self.mesh.edgestats()[2]

        near  = .9
        far   = 3.1
        right = 1.1 
        left  = hi * self.slicetolerance_end
        top   = hi * self.slicetolerance_halfwidth

        fl = Frustum().lrtbnf( ( -right, left, -top, top, near, far ) )
        fr = Frustum().lrtbnf( ( left,  right, -top, top, near, far ) )

        fl.orthographic()
        fr.orthographic()

        edgestep = hi * 1.35
        segs = int(math.ceil((math.pi)/edgestep))

        return fl, fr, segs

    def buildmesh( self, levels ):
        '''builds a list of triangle meshs starting with an icosadehron
        this base mesh is subdivided into many levels
        each lower level has 4 times as many faces
        subdivided points are normalized to lie on the surface of the sphere'''
        lpyramid = [TriMesh.icosa()]
        for i in xrange(0, levels):
            lnext = lpyramid[i].subdivide()
            lnext.normalizepoints()
            lpyramid.append(lnext)
            
        self.mesh = lpyramid[-1]
        self.mesh.name = 'sphere'

if __name__ == '__main__':

    def dump2gl( meshlist ):
        'save a list of meshes'
        print('saving result meshes:')
        args = []
        for mesh in meshlist:
            tmp = '/var/tmp/%s.obj' % (mesh.name)
            if mesh.uvs is None:
                mesh.uvzeros()
            mesh.writeobj(tmp)
            args.append(tmp)
            print tmp

    def seedarg():
        'get an integer seed command line argument'
        seed = 0
        try:
            import sys
            seed = int(sys.argv[1])
        except (IndexError, ValueError):
            print('cannot get a seed int from last argument')
        return seed

    def vecmesh( vec, name):
        'degenerate triangle describing vector +- around origin'
        return TriMesh( [[0, 0, 0], vec*-2, vec*2], [0, 1, 2], name=name)

    def randfrustums( n, scl, solver, squish=None, seedstart=0 ):
        'randomly distributed view frustums'
        mindimension = solver.mesh.edgestats()[2]
        result = []
        for i in xrange(n):
            f = Frustum.rand( scl, i + seedstart, mindimension, squish)
            result.append( f )
        return result

    def run( solver ):
        'wrapper for cProfile'
        return solver.run()

    def testmulti( solver_kind = SeamSolver.kfree): #kfree ): #kzup ):
        '''computes frustum/mesh intersection and other stuff
        build a pyramid, build frustums and find seams'''
        levels       = 5
        nfrustums    = 25
        frustumscale = .4
        squishscale  = .1
        seed         = seedarg()

        solver = SeamSolver( levels, solver_kind )
    
        meshlist = [solver.mesh]

        squish = numpy.identity(4)

        if solver_kind == SeamSolver.kplane:
            squishvec = num.randunitpt( seed )
            squish = num.vecspace( squishvec  )
            meshlist.append( vecmesh( squishvec, 'squish' ) )
        if solver_kind in (SeamSolver.kzup, SeamSolver.kplane):
            squish[2, :] *= squishscale

        frustums = randfrustums( nfrustums, 
                                 frustumscale,
                                 solver, 
                                 squish,
                                 seed )

        solver.markfrustums(frustums)

        solver.mesh.uvzeros()
        solver.mesh.uvs[:, 0] = 1-solver.vertweight
        solver.mesh.uvs[:, 1] = solver.vertvis

        seams = solver.run()

        if seams:
            dummy, vis, f = seams[0]
            
            polevec = num.pointmatrixmult( num.Z, f.ctw )
            meshlist.append( vecmesh( polevec[0], 'pole' ) )

            seamslice = TriMesh.cube()
            seamslice.points[:, 0] *= .5
            seamslice.points[:, 0] -= .5
            seamslice.points[:, 0] *= 1.1
            seamslice.points[:, 1] *= solver.mesh.edgestats()[-1]
            seamslice.points[:, 1] *= solver.slicetolerance_halfwidth
            seamslice.points[:, 2] *= 1.1

            seamslice.xform( f.ctw )
            seamslice.name = 'seam'
            meshlist.append( seamslice )

            flat = solver.mesh.copy().polar(f)
            flat.name = 'unwrap'
            meshlist.append( flat )

            solver.mesh.uvs[vis] = 1, 1 
        else:
            print('no seams found')
   
       #dump2gl(meshlist)

    def testcells():
        ''' give grid cells random colors for debugging'''
        solver = SeamSolver()

        cells, ncells = solver.mesh.cells( 3 )
        colors = numpy.random.random( (ncells, 2) )
    
        solver.mesh.uvzeros()
        for i in range(ncells):
            verts = cells == i
            solver.mesh.uvs[verts] = colors[i]

        meshlist = [solver.mesh]

        dump2gl(meshlist)

    import cProfile
    cProfile.run('testmulti()')
   #testmulti()
   #testcells()
