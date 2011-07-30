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
from constants import DTYPE

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
        
def randunitpt(seed = None):
    'a random unit length vector'
    if not seed == None:
        random.seed(seed)
    v = numpy.array((random.uniform(-1, 1),
                     random.uniform(-1, 1),
                     random.uniform(-1, 1) ), dtype=DTYPE)
    return normalize(v)

def vecspace2( z, x ):
    'obect space matrix along a given z axis'
    z = normalize(z)
    x = normalize(x)
    y = normalize(numpy.cross( x, z ))
    x = normalize(numpy.cross( y, z ))

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
    'translate xform helper'
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
    return hpoints[:, :3]

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

def numpy2pointer(a):
    '''returns a ctypes compatable pointer to the numpy array
    most arrays will already be contiguousarray, so probably a no-op
    unfortunately numpy.ndarray.ctypes.data_as is undocumented
    '''

    match = { numpy.int64  :ctypes.c_int64,
              numpy.int32  :ctypes.c_int32,
              numpy.float32:ctypes.c_float,
              numpy.float64:ctypes.c_double }

    ptype =  ctypes.c_float
    if a.dtype == numpy.float64: 
        ptype =  ctypes.c_double
    if a.dtype == numpy.int32: 
        ptype =  ctypes.c_int32
    if a.dtype == numpy.int64: 
        ptype =  ctypes.c_int64

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

if __name__ == '__main__':
   
    p0 = vec3( 1,  1, 0) 
    p1 = vec3( 1, -1, 0)
    hits = sphere_line_intersection( p0, p1, vec3(), 1.0 )
    print(hits)
