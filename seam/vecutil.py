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
'helper functions for 3 space vector operations in numpy'

import numpy
import random
import ctypes
import math
from constants import DTYPE, INTDTYPE

def vec3( x = 0.0, y = 0.0, z = 0.0):
    'numpy array triple helper'
    return numpy.array( (x, y, z), dtype=DTYPE )

def mat4():
    'numpy matrix 4x4 helper'
    return numpy.identity(4, dtype=DTYPE )

X  = vec3(1, 0, 0)
Y  = vec3(0, 1, 0)
Z  = vec3(0, 0, 1)
SMALL = 0.0001
TINY  = 0.00000001

def resizevector( input, columns=4, fill=0.0 ):
    if not hasattr( input, 'shape'):
        input = numpy.array(input, dtype=DTYPE)
    oned = len(input.shape) == 1
    if oned:    
        input = input[numpy.newaxis]
    elif len(input.shape) != 2:
        raise Warning( 'needs 1 or 2 dimensions')
    rows = input.shape[0]

    result = numpy.zeros( (rows,columns), dtype=input.dtype ) + fill

    for i in range(min( input.shape[1], columns)):
        result[:,i] = input[:,i]

    if oned:
        result = result[0]

    return result

def asvec3( input, fill=0.0):
    return resizevector( input, columns=3, fill=fill )

def asvec4( input, fill=0.0):
    return resizevector( input, columns=4, fill=fill )

def vlength(v):
    'wraps numpy, does not include possible homogenious coordinate'
    return numpy.linalg.norm(v[0:3])

def normalize(v):
    'length should be 1, except for tiny vectors which are unchanged'
    length = vlength(v)
    if length > TINY:
        return v / length
    else:
        return v

def vlengths(v):
    x = v[:,0]
    y = v[:,1]
    z = v[:,2]
    return numpy.sqrt( x*x + y*y + z*z)
        
def normalizes(v):
    x = v[:,0]
    y = v[:,1]
    z = v[:,2]
    invscale = 1.0 / vlengths(v)
    x *= invscale
    y *= invscale
    z *= invscale

def randunitpts(n, seed = None):
    'random unit length vectors'
    if not seed == None:
        numpy.random.seed(seed)
    p = numpy.zeros( (n,3), dtype=DTYPE )
    p[:,0] = numpy.random.uniform( 0, 1, n )
    p[:,1] = numpy.random.uniform( 0, 1, n )
    p[:,2] = numpy.random.uniform( 0, 1, n )

    normalizes(p)
    return p

def randunitpt(seed = None):
    'a random unit length vector'
    if not seed == None:
        random.seed(seed)
    v = numpy.array((random.uniform(-1, 1),
                     random.uniform(-1, 1),
                     random.uniform(-1, 1) ), dtype=DTYPE)
    return normalize(v)

def handlesFromMx( mx ):
    'z and x from a matrix'
    z = normalize(mx[2, :3])
    y = normalize(mx[1, :3])
    x = normalize(numpy.cross( y, z ))

    righthanded = numpy.linalg.det( mx ) > 0

    return z, x, righthanded, mx[3,:3]

def mxFromHandles( z, x, righthanded, pos ):
    mx = vecspace2( z, x, righthanded )
    mx[3,:3] = pos
    return mx

def mxFromXYQuad( points ):
    '3 point affine mapping'
    p0 = points[0]
    p1 = points[1]
    p2 = points[3]

    result = numpy.identity(4, dtype=points.dtype)
    result[0,:3] = p1 - p0
    result[1,:3] = p2 - p0
    result[3,:3] = p0
    return result

def mxFromXYQuadPerspective(points):
    '''4 point corner pin mapping
     sovle A * H = 0, where A = [  0  -w*p  y*p ]
                                [ w*p   0  -x*p ]
                                [-y*p  x*p   0  ]
    xp = unit square * p.x
    yp = unit square * p.y
    wp = unit square  ( p.w is == 1 )
    p is a list of 4 points

    can be generalized to any n points -- restrict to quads for now.

    H is gven by the last column of V where
    A = UEVt is the signular value decomposition of A

    full explanation:
    http://www.cse.iitd.ac.in/~suban/vision/geometry/node24.html

    serial code:
    https://github.com/relet/Holodeck-Minigolf/blob/master/homography.py
'''
    wp = numpy.array([[ 0, 0, 1],
                      [ 1, 0, 1],
                      [ 1, 1, 1],
                      [ 0, 1, 1]]).astype( 'f8')

    xp, yp = wp.copy(), wp.copy()
    xp[:,0] *= points[:,0]
    xp[:,1] *= points[:,0]
    xp[:,2] *= points[:,0]

    yp[:,0] *= points[:,1]
    yp[:,1] *= points[:,1]
    yp[:,2] *= points[:,1]

    A = numpy.zeros((12,9),dtype='f8')
    A[0::3, 3:6] = -wp
    A[0::3, 6:9] =  yp
    A[1::3, 0:3] =  wp
    A[1::3, 6:9] = -xp
    A[2::3, 0:3] = -yp
    A[2::3, 3:6] =  xp

    U, e, V      = numpy.linalg.svd(A, full_matrices=False)

    mx33         = V[-1].reshape(3,3).T
    mx44         = mat4()
    mx44[:2,:2]  = mx33[:2,:2]
    mx44[3,:2]   = mx33[ 2,:2]
    mx44[:2,3]   = mx33[:2, 2]
    mx44[3,3]    = mx33[ 2, 2]
    mx44[2,2]    = 0
    return mx44


def vecspace2( z, x, righthanded=True ):
    'obect space matrix along a given z axis'
    z = normalize(z)
    x = normalize(x)
    y = normalize(numpy.cross( x, z ))
    x = normalize(numpy.cross( y, z ))

    if not righthanded:
        y *= -1

    mx = numpy.zeros((4, 4), dtype=DTYPE)
    mx[0, :3] = x
    mx[1, :3] = y
    mx[2, :3] = z
    mx[3,  3] = 1.0

    return mx

def vecspace( z ):
    'obect space matrix along a given z axis'
    z = normalize(z)
    minaxisidx = numpy.argmin( numpy.abs(z) )
    minaxis    = (X, Y, Z)[minaxisidx]
    y = normalize(numpy.cross( minaxis, z ))
    x = normalize(numpy.cross( y, z ))

    mx = numpy.zeros((4, 4), dtype=DTYPE)
    mx[0, :3] = x
    mx[1, :3] = y
    mx[2, :3] = z
    mx[3,  3] = 1.0

    return mx

def scalematrix( x = 0.0, y = 0.0, z = 0.0):
    'scale xform helper'
    m = numpy.array( [[ x , 0., 0., 0 ]
                     ,[ 0., y , 0., 0 ]
                     ,[ 0., 0., z , 0 ]
                     ,[ 0., 0., 0., 1.]], dtype=DTYPE)
    return m
 
def translatematrix( x = 0.0, y = 0.0, z = 0.0):
    'translate xform helper'
    m = numpy.array( [[ 1 , 0., 0., 0 ]
                     ,[ 0., 1 , 0., 0 ]
                     ,[ 0., 0., 1 , 0 ]
                     ,[ x , y , z , 1.]], dtype=DTYPE)
    return m

def xrotatematrix( t, degrees = False):
    'rotate around x axis in radians or degrees'
    if degrees:     
        t = math.radians(t)

    c, s = math.cos(t), math.sin(t)
    r = numpy.array([[  1,  0,  0,  0], 
                     [  0,  c,  s,  0], 
                     [  0, -s,  c,  0],
                     [  0,  0,  0,  1]], dtype=DTYPE)
    return r

def yrotatematrix( t, degrees = False):
    'rotate around y axis in radians or degrees'
    if degrees:     
        t = math.radians(t)

    c, s = math.cos(t), math.sin(t)
    r = numpy.array([[  c,  0, -s,  0], 
                     [  0,  1,  0,  0], 
                     [  s,  0,  c,  0],
                     [  0,  0,  0,  1]], dtype=DTYPE)
    return r

def zrotatematrix( t, degrees = False):
    'rotate around z axis in radians or degrees'
    if degrees:     
        t = math.radians(t)

    c, s = math.cos(t), math.sin(t)
    r = numpy.array([[  c,  s,  0,  0], 
                     [ -s,  c,  0,  0], 
                     [  0,  0,  1,  0],
                     [  0,  0,  0,  1]], dtype=DTYPE)
    return r


def rotate(pts, angle, axis, degrees = False):
    'rotate points by angle, around axis x, y or z'
    matrix = numpy.identity(4)
    if axis == 'x':
        matrix = xrotatematrix(angle, degrees)        
    elif axis == 'y':
        matrix = yrotatematrix(angle, degrees)        
    elif axis == 'z':
        matrix = zrotatematrix(angle, degrees)
    return numpy.dot(pts, matrix)

def translate(pts, x=0.0, y=0.0, z=0.0):
    'translate points'
    matrix = translatematrix(x, y, z)
    return numpy.dot(pts, matrix)

def scale(pts, x=1.0, y=1.0, z=1.0):
    'scale points'
    matrix = scalematrix(x, y, z)
    return numpy.dot(pts, matrix)


def raisedividebyzero():
    numpy.seterr( divide='raise' )

def warndividebyzero():
    numpy.seterr( divide='warn' )

def pointmatrixmult( point, mx ):
    'homogenious point matrix multiply, return result'
    if len(point.shape) == 1:
        hpoints = numpy.ones( (1, 4), dtype=DTYPE)
    else:
        hpoints = numpy.ones( (len(point), 4), dtype=DTYPE)
    hpoints[:, :3] = point
    hpoints = numpy.dot( hpoints, mx)
    wcoords = hpoints[:, 3]
    hpoints[:, 0] /= wcoords
    hpoints[:, 1] /= wcoords
    hpoints[:, 2] /= wcoords
    npoints = hpoints[:, :3]
    return numpy.ascontiguousarray(npoints)

def polar2xyz( points ):
    'transform points to cartesian'
    ruv = points.copy()
    r = ruv[:, 0]
    u = ruv[:, 1]
    v = ruv[:, 2]

    z         = numpy.sin( v ) * r
    ringscale = numpy.cos( v )
    x         = ringscale * numpy.cos( u ) * r
    y         = ringscale * numpy.sin( u ) * r

    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = z

def xyz2polar( points ):
    'transform points to polar'
    xyz = points.copy()
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    r = numpy.sqrt( x*x + y*y + z*z)
    x /= r
    y /= r
    z /= r

    v         = numpy.arcsin( z )
    ringscale = numpy.cos( v )
    u         = numpy.arctan2( y / ringscale, x / ringscale )

    points[:, 0] = r
    points[:, 1] = u
    points[:, 2] = v



#   ntype2ctype = {numpy.float64 : ctypes.c_double
#                 ,numpy.uint32  : ctypes.c_uint32
#                 ,numpy.int32   : ctypes.c_int32
#                 ,numpy.uint64  : ctypes.c_uint64
#                 ,numpy.int64   : ctypes.c_int64
#                 ,numpy.int8    : ctypes.c_int8 }

def ntype2ctype(n):
    if isinstance(n,numpy.ndarray):
        n = n.dtype

    ptype =  ctypes.c_float
    if n == numpy.float64: 
        ptype =  ctypes.c_double
    if n == numpy.uint32: 
        ptype =  ctypes.c_uint32
    if n == numpy.int32: 
        ptype =  ctypes.c_int32
    if n == numpy.uint64: 
        ptype =  ctypes.c_uint64
    if n == numpy.int64: 
        ptype =  ctypes.c_int64
    if n == numpy.int8: 
        ptype =  ctypes.c_int8

    return ptype

def numpy2pointerref(a):
    '''returns a ctypes compatable pointer to the numpy array
    most arrays will already be contiguousarray, so probably a no-op
    unfortunately numpy.ndarray.ctypes.data_as is undocumented
    return the contiguous array to avoid garbage collection
    '''

    ptype = ntype2ctype( a )
#   print( a.dtype, ptype)
#   ptype =  ctypes.c_float
#   if a.dtype in ntype2ctype
#   if a.dtype == numpy.float64: 
#       ptype =  ctypes.c_double
#   if a.dtype == numpy.uint32: 
#       ptype =  ctypes.c_uint32
#   if a.dtype == numpy.int32: 
#       ptype =  ctypes.c_int32
#   if a.dtype == numpy.uint64: 
#       ptype =  ctypes.c_uint64
#   if a.dtype == numpy.int64: 
#       ptype =  ctypes.c_int64
#   if a.dtype == numpy.int8: 
#       ptype =  ctypes.c_int8

    na = numpy.ascontiguousarray(a)
    return na.ctypes.data_as(ctypes.POINTER(ptype)), na


def numpy2pointer(a):
    '''returns a ctypes compatable pointer to the numpy array
    most arrays will already be contiguousarray, so probably a no-op
    unfortunately numpy.ndarray.ctypes.data_as is undocumented
    '''

    ptype =  ctypes.c_float
    if a.dtype == numpy.float64: 
        ptype =  ctypes.c_double
    if a.dtype == numpy.uint32: 
        ptype =  ctypes.c_uint32
    if a.dtype == numpy.int32: 
        ptype =  ctypes.c_int32
    if a.dtype == numpy.uint64: 
        ptype =  ctypes.c_uint64
    if a.dtype == numpy.int64: 
        ptype =  ctypes.c_int64
    if a.dtype == numpy.int8: 
        ptype =  ctypes.c_int8

    a = numpy.ascontiguousarray(a)
    return a.ctypes.data_as(ctypes.POINTER(ptype))

def unwrap1point(mesh, orientmx, distantvert):
    '''unwrap mesh using arc distance from z axis in input space'''
    unwrap_pts = mesh.xformresult( numpy.linalg.inv(orientmx) )

    antipodes = mesh.antipodes(distantvert)
    
    lengths = numpy.arccos( numpy.dot( unwrap_pts, Z) )
    lengths[distantvert] = 0
    lengths[antipodes]   = math.pi * 2

    x = unwrap_pts[:, 0]
    y = unwrap_pts[:, 1]
    xylens = numpy.sqrt(x*x + y*y)
    xylens = 1.0  / xylens

    xylens[distantvert] = 0
    xylens[antipodes]   = 0
    
    unwrap_pts[:, 0] = x * xylens * lengths
    unwrap_pts[:, 1] = y * xylens * lengths
    unwrap_pts[:, 2] = 1

    hi = mesh.edgestats()[2]

    othersideidxs = lengths > math.pi - hi * 3
    unwrap_pts[othersideidxs, 2] = 0
    return unwrap_pts, lengths

def resampleuniform( x, y, step ):
    '''input x/y pairs where the x values are unevenly spaced
    fit a polynomial to x/y pairs
    creates a uniform grid in x: ux
    evaluates polynomial on ux grid: uy'''
    ncoef = 25
    coef, dummy1, rank, dummy2, dummy3 = \
        numpy.polyfit( x, y, ncoef, full=True )
    if rank != ncoef + 1:
        coef, dummy1, dummy2, dummy3, dummy4 = \
            numpy.polyfit( x, y, rank, full=True )

    ux = numpy.arange( x[0], x[-1], step )
    uy = numpy.polyval( coef, ux)

    return ux, uy

def minspansearch( x, y ):
    '''1d search along x/y pairs is hard for uneven spacing in x
    resample input curve to be uniform in x
    convolve with a window of length pi: ystar
    min value in ystar corresponds to the best span
    ystar is a different length than uy/ux
    find the ux coord that corresponds to the best span
    find the x coords near span - 90 deg, span + 90 deg
    '''
    pisteps = 180
    ux, uy = resampleuniform( x, y, math.pi / pisteps )

    kernel = numpy.ones( pisteps, dtype=DTYPE )

    ystar = numpy.convolve( uy, kernel, 'valid')
    pivot = numpy.argmin( ystar )
    spanweight = ystar[pivot]

    nxmid     = int(round(.5*(len(ux))))
    nystarmid = int(round(.5*(len(ystar))))
    shift     = nxmid - nystarmid
    pivotx    = ux[pivot + shift]

    pivot = numpy.argmin( numpy.abs( x - pivotx ) )
    loval = x[pivot] - (math.pi / 2)
    hival = x[pivot] + (math.pi / 2)
    lo = numpy.argmin( numpy.abs( x - loval ) )
    hi = numpy.argmin( numpy.abs( x - hival ) )

    return spanweight, lo, pivot, hi

def sphere_line_intersection(l1, l2, sp, r):
    ''' takes line start, line stop, sphere center and radius
    reference: http://paulbourke.net/geometry/sphereline/
    returns a 2-tuple of Nones for a miss
    1 vec3 and 1 None for a tangent hit
    2 vec3 for a double hit
    '''
    p1 = p2 = None

    v = l2 - l1

    a = numpy.sum( v**2 )

    b = 2 * numpy.sum( v * ( l1 - sp ) )

    c = numpy.sum( sp**2 + l1**2) - 2 * numpy.sum(sp * l1) - r**2

    i = b * b - 4.0 * a * c

    if i < 0.0:
        pass  # no intersections
    elif i == 0.0:
        # one intersection
        mu = -b / (2.0 * a)
        p1 = l1 + v * mu

    elif i > 0.0:
        rooti = math.sqrt(i)
        # first intersection
        mu = (-b + rooti) / (2.0 * a)
        p1 = l1 + v * mu

        # second intersection
        mu = (-b - rooti) / (2.0 * a)
        p2 = l1 + v * mu

    return p1, p2

def matrixstackmultfull(a,b):
    result = numpy.zeros(a.shape)
    for i in range(0,4):
        for j in range(0,4):
            for k in range(0,4):
                result[i,j] += a[i,k] * b[k,j]

    return result

def matrixstackmult(a,b):
    result = numpy.zeros(a.shape)
    for i in range(0,4):
        for j in range(0,4):
            result[:,i,j] = numpy.sum( a[:,i,:] * b[:,:,j], axis=1)
           #for k in range(0,4):
           #    result[i,j] += a[i,k] * b[k,j]

    return result

def revolve_bspline_fast( points, n_verts_u, n_verts_v, start, end):
    '''optimized version of revolve_bspline_slow
    makes a bspline curve / surface of revolution
    sweeps points out around the z axis
    like all bsplines, verts[0] and verts[-1] are outside the sweep range
    '''
    n_spans = n_verts_u - 3
    assert( n_verts_v == len(points) )
    assert( n_spans   > 0 )

        #scalars
    full_angle  = end - start
    delta_angle = full_angle / n_spans
    radius      = 3./(2. + numpy.cos(delta_angle))

        #vectors of length n_verts_u
    indexes     = numpy.arange( n_verts_u )
    steps       = (indexes-1) * delta_angle
    angles      = start + steps
    cos_scale   = radius * numpy.cos(angles)
    sin_scale   = radius * numpy.sin(angles)

        #grids of shape [n_verts_u*n_verts_v, 4]
    input_grid  = points.repeat( n_verts_u, axis=0 )
    result      = input_grid.copy()

        #vectors of length n_verts_u*n_verts_v
    x, y        = input_grid[:,0], input_grid[:,1]
    cos_scale   = cos_scale[numpy.newaxis].repeat(n_verts_v,axis=0).ravel()
    sin_scale   = sin_scale[numpy.newaxis].repeat(n_verts_v,axis=0).ravel()
        
                  #rotate and scale x,y points into swept position
    result[:,0] = cos_scale * x + ( -1 * sin_scale * y )
    result[:,1] = cos_scale * y + (      sin_scale * x )
  
    if n_verts_v != 1:
        result = result.reshape( (n_verts_v,n_verts_u,4) )
    return result

def revolve_bspline_slow( points, n_verts_u, n_verts_v, start, end):
    '''straightforward method to make a bspline curve / surface of revolution
    sweeps points out around the z axis
    like all bsplines, verts[0] and verts[-1] are outside the sweep range
    '''

    n_spans     = n_verts_u - 3
    full_angle  = end - start
    delta_angle = full_angle / n_spans
    radius      = 3./(2. + numpy.cos(delta_angle))

    result = []
    for index in range(n_verts_u):
        step = (index-1) * delta_angle
        current_angle = start + step

        cos_scale = radius * numpy.cos(current_angle)
        sin_scale = radius * numpy.sin(current_angle)

        npoints = points.copy()

        npoints[:,0] = cos_scale * points[:,0] + ( -1 * sin_scale * points[:,1] )
        npoints[:,1] = cos_scale * points[:,1] + (      sin_scale * points[:,0] )

        result.append( npoints )

    if n_verts_v == 1:
        return numpy.array( result )[:,0,:]

    return numpy.array( result).transpose( 1, 0, 2)

if __name__ == '__main__':
    pass
#   v_div = 10
#   vlo = -.1
#   vhi = .1
#   circ = bSplineSweep(numpy.array([1., 0., 0., 1.], dtype=numpy.float64), v_div, (vlo, vhi))
#   print( circ ) 

#   u_faces = 22
#   v_faces = 12
#   u_verts = u_faces + 1
#   v_verts = v_faces + 1
#   r0 = meshtest()
#   r = grid_indexes( u_verts, v_verts )
#   r = grid_indexes( (u_verts, v_verts) )
#   print(r)
#   print( (r0 == r.ravel()).all() )

#   indexes = numpy.arange( v_verts * u_verts ).reshape( (v_verts, u_verts) )

#   v0 = indexes[  :-1,  :-1]
#   v1 = indexes[  :-1, 1:  ]
#   v2 = indexes[ 1:  , 1:  ]
#   v3 = indexes[ 1:  ,  :-1]
#   print( v3 )
   #basis = numpy.mgrid[:5, :4]
   #print( basis[1,:,:] + ( basis[0,:,:]*4))

#   print( ( meshindexes( 22, 12 ) == meshtest()).all()  )
    
#   import frustum
#   n = 10
#   a = numpy.mgrid[:16].astype('f8')
#   a = numpy.tile( a, n )
#   a = a.reshape( (n,4,4) )

#   d = numpy.dot( a[0], a[0] )
#   h = matrixstackmult(a,a)

#   print(numpy.allclose(d,h[0]))
