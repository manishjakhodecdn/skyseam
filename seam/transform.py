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
'numpy based Transform class for seam'
import numpy
import vecutil
import constants

class Transform( object ):
    '''compose translate rotate scale as 4x4 matrix
    access object to world and world to object xforms as properties
    wraps numpy for all linear algebra'''
    ktranslate = 0
    kxrotate   = 1
    kyrotate   = 2
    kzrotate   = 3
    kscale     = 4
             
    def __init__( self, other=None ):
        'defaults: translate, rotate and scale'
        self._tx  = 0
        self._ty  = 0
        self._tz  = 0
        self._rx  = 0
        self._ry  = 0
        self._rz  = 0
        self._sx  = 1
        self._sy  = 1
        self._sz  = 1
        self._wto = numpy.identity(4,dtype=constants.DTYPE)
        self._otw = numpy.identity(4,dtype=constants.DTYPE)

        self._order = ( self.ktranslate
                       ,self.kscale
                       ,self.kzrotate
                       ,self.kyrotate
                       ,self.kxrotate)

        self._dirty = False

        if other is not None:
            self._tx  = other._tx
            self._ty  = other._ty
            self._tz  = other._tz
            self._rx  = other._rx
            self._ry  = other._ry
            self._rz  = other._rz
            self._sx  = other._sx
            self._sy  = other._sy
            self._sz  = other._sz
            self._wto = other._wto
            self._otw = other._otw
            self._order = other._order
            self._dirty = other._dirty 

    def translatematrix( self ):
        'uncached translate 4x4'
        return vecutil.translatematrix( self._tx, self._ty, self._tz )

    def xrotatematrix( self ):
        'uncached rx 4x4'
        return vecutil.xrotatematrix( self._rx )

    def yrotatematrix( self ):
        'uncached ry 4x4'
        return vecutil.yrotatematrix( self._ry )

    def zrotatematrix( self ):
        'uncached rz 4x4'
        return vecutil.zrotatematrix( self._rz )

    def scalematrix( self ):
        'uncached scale 4x4'
        return vecutil.scalematrix( self._sx, self._sy, self._sz )

    def _update( self ):
        'compose object space matrix'
        if self._dirty:
            dispatch = {self.ktranslate:self.translatematrix
                       ,self.kxrotate:self.xrotatematrix
                       ,self.kyrotate:self.yrotatematrix
                       ,self.kzrotate:self.zrotatematrix
                       ,self.kscale:self.scalematrix }
            self._otw = numpy.identity(4,dtype=constants.DTYPE)
            for i in self._order:
                self._otw = numpy.dot( self._otw, dispatch[i]() )

            self._wto = numpy.linalg.inv(self._otw)
            self._dirty = False

    def __mul__( self, other ):
        'Transform() * Transform() object to world matrix multiplication'
        return numpy.dot( self.otw, other.otw ) 

    @property
    def wto(self):
        'world to object matrix'
        self._update()
        return self._wto

    @property
    def t(self):
        'translation as a 3 vec'
        return vecutil.vec3( self.tx, self.ty, self.tz )

    @property
    def r(self):
        'rotation as a 3 vec'
        return vecutil.vec3( self.rx, self.ry, self.rz )

    @property
    def s(self):
        'scale as a 3 vec'
        return vecutil.vec3( self.sx, self.sy, self.sz )

    @property
    def otw(self):
        'object to world matrix'
        self._update()
        return self._otw

    @property
    def rx(self):
        'rotate x'
        return self._rx
    @rx.setter
    def rx(self, value):
        self._dirty = True
        self._rx = value

    @property
    def ry(self):
        'rotate y'
        return self._ry
    @ry.setter
    def ry(self, value):
        self._dirty = True
        self._ry = value

    @property
    def rz(self):
        'rotate z'
        return self._rz
    @rz.setter
    def rz(self, value):
        self._dirty = True
        self._rz = value

    @property
    def tx(self):
        'translate x'
        return self._tx
    @tx.setter
    def tx(self, value):
        self._dirty = True
        self._tx = value

    @property
    def ty(self):
        'translate y'
        return self._ty
    @ty.setter
    def ty(self, value):
        self._dirty = True
        self._ty = value

    @property
    def tz(self):
        'translate z'
        return self._tz
    @tz.setter
    def tz(self, value):
        self._dirty = True
        self._tz = value

    @property
    def sx(self):
        'scale x'
        return self._sx
    @sx.setter
    def sx(self, value):
        self._dirty = True
        self._sx = value

    @property
    def sy(self):
        'scale y'
        return self._sy
    @sy.setter
    def sy(self, value):
        self._dirty = True
        self._sy = value

    @property
    def sz(self):
        'scale z'
        return self._sz
    @sz.setter
    def sz(self, value):
        self._dirty = True
        self._sz = value

    def __str__(self):
        'Transform in string form'
        parts = []
        for name, mx in (('otw', self.otw)
                        ,('wto', self.wto)):

            if mx is None:
                parts.append( '%s:%s'%(name, None) )
            else:
                parts.append( '%s:\n%s'%(name, repr(mx)) )

                            
        for value, name in ((self.rx     ,'rotate x'     )
                           ,(self.ry     ,'rotate y'     )
                           ,(self.rz     ,'rotate z'     )
                           ,(self.tx     ,'translate x'  )
                           ,(self.ty     ,'translate y'  )
                           ,(self.tz     ,'translate z'  )
                           ,(self.sx     ,'scale x'      )
                           ,(self.sy     ,'scale y'      )
                           ,(self.sz     ,'scale z'      )):

            parts.append( '%s:%s'%(name, str(value)))
    
        return '\n'.join( parts )

__all__ = ['Transform']

if __name__ == '__main__':
    def _printproperties():
        'template code for the xform class'
        fmt = '''
        @property
        def %s(self):
            '%s'
            return self._%s
        @%s.setter
        def %s(self, value):
            self._dirty = True
            self._%s = value'''
        
        p = (('wto'    ,'world to object matrix' )
            ,('otw'    ,'object to world matrix' )
            ,('rx'     ,'rotate x'               )
            ,('ry'     ,'rotate y'               )
            ,('rz'     ,'rotate z'               )
            ,('tx'     ,'translate x'            )
            ,('ty'     ,'translate y'            )
            ,('tz'     ,'translate z'            )
            ,('sx'     ,'scale x'                )
            ,('sy'     ,'scale y'                )
            ,('sz'     ,'scale z'                ))
        for v, d in p:
            print(fmt% (v, d, v, v, v, v))
    def _transformtest():
        'test transform class'
        a = Transform()
        a.tx = 1
       #a.sx = a.sy = a.sz = 10
       #a.rx = a.ry = a.rz = .707

        b = Transform()
        print(b.otw.dtype)
        b.tx = 1
        r = a * b
        print(r)
        print(r.dtype)

    _transformtest()
   #_printproperties()
