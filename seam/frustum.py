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
'camera frustum module'

import numpy
import random
import vecutil
import constants

__all__ = ['Frustum']
class Frustum(object):
    'general camera model'
    def __init__( self):
        'defaults:worldToScreen, worldToCamera, inverses, etc.'
        self._wts    = None
        self._stw    = None

        self._cts    = None
        self._stc    = None

        self._wtc    = None
        self._ctw    = None

        self._left   = None
        self._right  = None
        self._top    = None
        self._bottom = None
        self._near   = None
        self._far    = None

        self._kind   = None

    def _worldfromobject(self):
        '''compose screenToCamera and cameraToWorld into screenToWorld
        invert to get worldToScreen'''
        if None not in ( self.stc, self.ctw ):
            self._stw = numpy.dot( self.stc, self.ctw )
            self._wts = numpy.linalg.inv(self._stw)

    @property
    def wts(self):
        'world to screen matrix'
        if self._wts is None and self._stw is not None:
            self._wts = numpy.linalg.inv(self._stw)
        else:
            self._worldfromobject()
        return self._wts

    @wts.setter
    def wts(self, value):
        self._wts = value

    @property
    def stw(self):
        'screen to world matrix'
        if self._stw is None and self.wts is not None:
            self._stw = numpy.linalg.inv(self.wts)
        return self._stw
    @stw.setter
    def stw(self, value):
        self._stw = value

    @property
    def cts(self):
        'camera to screen matrix'
        if self._cts is None and self._stc is not None:
            self._cts = numpy.linalg.inv(self._stc)
        return self._cts
    @cts.setter
    def cts(self, value):
        self._cts = value

    @property
    def stc(self):
        'screen to camera matrix'
        if self._stc is None and self._cts is not None:
            self._stc = numpy.linalg.inv(self._cts)
        return self._stc
    @stc.setter
    def stc(self, value):
        self._stc = value

    @property
    def wtc(self):
        'world to camera matrix'
        if self._wtc is None and self._ctw is not None:
            self._wtc = numpy.linalg.inv(self._ctw)
        return self._wtc
    @wtc.setter
    def wtc(self, value):
        self._wtc = value

    @property
    def ctw(self):
        'camera to world matrix'
        if self._ctw is None and self._wtc is not None:
            self._ctw = numpy.linalg.inv(self._wtc)
        return self._ctw
    @ctw.setter
    def ctw(self, value):
        self._ctw = value

    @property
    def left(self):
        'left clip plane'
        if self._left is None and self._right is not None:
            self.left = -self._right
        return self._left
    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        'right clip plane'
        return self._right
    @right.setter
    def right(self, value):
        self._right = value

    @property
    def top(self):
        'top clip plane'
        return self._top
    @top.setter
    def top(self, value):
        self._top = value

    @property
    def bottom(self):
        'bottom clip plane'
        if self._bottom is None and self._top is not None:
            self.bottom = -self._top
        return self._bottom
    @bottom.setter
    def bottom(self, value):
        self._bottom = value

    @property
    def near(self):
        'near clip plane'
        return self._near
    @near.setter
    def near(self, value):
        self._near = value

    @property
    def far(self):
        'far clip plane'
        return self._far
    @far.setter
    def far(self, value):
        self._far = value

    @property
    def kind(self):
        'kind of projection: ortho|persp'
        return self._kind
    @kind.setter
    def kind(self, value):
        self._kind = value
        
    def rtnf( self, rtnf):
        'set params symmetric right/left top/bottom'
       
        self.right = rtnf[0]
        self.top   = rtnf[1]
        self.near  = rtnf[2]
        self.far   = rtnf[3]

        self.left   = - self.right
        self.bottom = - self.top

        return self

    def lrtbnf( self, lrtbnf):
        'set params'
        self.left   = lrtbnf[0]
        self.right  = lrtbnf[1]
        self.top    = lrtbnf[2]
        self.bottom = lrtbnf[3]
        self.near   = lrtbnf[4]
        self.far    = lrtbnf[5]
        return self

    def perspective(self):
        'general gl style perspective projection matrix'
        x = (2.0 * self.near)  / (self.right - self.left)
        y = (2.0 * self.near)  / (self.top - self.bottom)

        a = (self.right + self.left) / (self.right - self.left)
        b = (self.top + self.bottom) / (self.top - self.bottom)
        c = (self.far + self.near)   / (self.far - self.near)

        d = (2 * self.far * self.near) / (self.far - self.near)
        c *= -1
        d *= -1

        self.kind = 'persp'

        self.cts = numpy.array( [[ x , 0.,  a , 0.] 
                                ,[ 0., y ,  b , 0.] 
                                ,[ 0., 0.,  c , d ] 
                                ,[ 0., 0., -1., 0.]]
                                ,dtype=constants.DTYPE).transpose()

        return self

    def orthographic(self):
        'general gl style orthographic projection matrix'
        x = 2.0 / (self.right-self.left)
        y = 2.0 / (self.top-self.bottom)
        z = 2.0 / (self.far-self.near)
        a = (self.right+self.left) / (self.right-self.left)
        b = (self.top+self.bottom) / (self.top-self.bottom)
        c = (self.far+self.near)   / (self.far-self.near)
        z *= -1
        a *= -1
        b *= -1
        c *= -1

        self.kind = 'ortho'

        self.cts = numpy.array( [[ x , 0., 0., a ]
                                ,[ 0., y , 0., b ]
                                ,[ 0., 0., z , c ]
                                ,[ 0., 0., 0., 1.]]
                                ,dtype=constants.DTYPE).transpose()

        return self

    @staticmethod
    def rand( scl, seed, mindimension, squish=None ):
        'random frustum'
        random.seed(seed)
   

        z = vecutil.randunitpt()

        if squish is not None:
            z = vecutil.pointmatrixmult( z, squish )
            z = vecutil.normalize(z)

        y = vecutil.randunitpt()
        x = vecutil.normalize(numpy.cross(z, y))
        y = vecutil.normalize(numpy.cross(x, z))
        
        siz =  vecutil.randunitpt()*scl

        ctw = numpy.zeros((4, 4))
        ctw[0, :3] = x
        ctw[1, :3] = y
        ctw[2, :3] = -z
        ctw[3,  3] = 1.0
       
        f = Frustum()
        f.ctw   = ctw 
        f.far   = 2
        f.near  = .1
        f.right = max(abs(siz[0]), mindimension*.75)*f.near
        f.top   = max(abs(siz[1]), mindimension*.75)*f.near
        f.perspective()

        return f

    def __str__(self):
        'frustum in string form'
        parts = []
        for name, mx in (('cts', self.cts)
                        ,('stc', self.stc)
                        ,('wts', self.wts)
                        ,('stw', self.stw)
                        ,('wtc', self.wtc)
                        ,('ctw', self.ctw)):

            if mx is None:
                parts.append( '\n  %s:%s'%(name, None) )
            else:
                parts.append( '\n  %s:%s'%(name, repr(mx)[6:-1]) )

                            
        for name, value in ((' left' , self.left   )
                           ,(' right' , self.right  )
                           ,(' top' , self.top    )
                           ,(' bottom' , self.bottom )
                           ,(' near' , self.near   )
                           ,(' far' , self.far    )
                           ,(' kind' , self.kind   )):
            parts.append( '%s:%s'%(name, str(value)))
    
        return 'Frustum(%s)' % ','.join( parts )

    def copy(self):
        'use deepcopy to copy all numpy and scalar members'
        def copyhelper( val ):
            'copy numpy array if possible'
            if val is None: return None
            return val.copy()
        f = Frustum()
        f._wts    = copyhelper( self._wts )
        f._stw    = copyhelper( self._stw )
        f._cts    = copyhelper( self._cts )
        f._stc    = copyhelper( self._stc )
        f._wtc    = copyhelper( self._wtc )
        f._ctw    = copyhelper( self._ctw )
        f._left   = self._left   
        f._right  = self._right  
        f._top    = self._top    
        f._bottom = self._bottom 
        f._near   = self._near   
        f._far    = self._far    
        f._kind   = self._kind
        return f

if __name__ == '__main__':
    def _frustumtest():
        'test frustum class'
        a = Frustum()
        a.rand(.6, 1, .01)

        a.stw = numpy.identity(4)
        print( a.wts )

        f = Frustum().rtnf((.1, .2, 3, 4)).orthographic()
        print(f)
        nf = Frustum().lrtbnf((.1, -.05, .2, -.15,  1, 2)).perspective()
        print(nf)

        of = nf.copy()
        nf.wtc = numpy.identity(4)
        of.wtc = nf.wtc * 5
        print(of)

    def _printproperties():
        'template code for the Frustum class'
        fmt = '''
        @property
        def %s(self):
            '%s'
            return self._%s
        @%s.setter
        def %s(self, value):
            self._%s = value'''
        
        p = (('wts'     ,'world to screen matrix'           )
            ,('stw'     ,'screen to world matrix'           )
            ,('cts'     ,'camera to screen matrix'          )
            ,('stc'     ,'screen to camera matrix'          )
            ,('wtc'     ,'world to camera matrix'           )
            ,('ctw'     ,'camera to world matrix'           )
            ,('left'    ,'left clip plane'                  )
            ,('right'   ,'right clip plane'                 )
            ,('top'     ,'top clip plane'                   )
            ,('bottom'  ,'bottom clip plane'                )
            ,('near'    ,'near clip plane'                  )
            ,('far'     ,'far clip plane'                   )
            ,('kind'    ,'kind of projection: ortho|persp'  ))

        for v, d in p:
            print(fmt% (v, d, v, v, v, v))
    _frustumtest()
    _printproperties()
