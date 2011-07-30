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
'simple opengl / pyglet based viewer for seam'

from pyglet.gl import *
import pyglet
from pyglet.window import key
import numpy
import ctypes
import controls
import math
import vecutil
from solver import SeamSolver
import constants
from trimesh import TriMesh
from transform import Transform
from frustum import Frustum


def glMultMatrix( mx ):
    'transform the gl stream'
    mxptr = vecutil.numpy2pointer( mx )
    size = mx.dtype.itemsize

    if size not in (4,8):
        raise Warning('matrix data must be 4 or 8 bytes')
    if size == 8:
        pyglet.gl.glMultMatrixd( mxptr )
    else:
        pyglet.gl.glMultMatrixf( mxptr )

class TriMeshGL( object ):
    kxyz = 'kxyz'
    kruv = 'kruv'
    def __init__(self, mesh, space=kxyz, colors=None, ui=None ):
        self.mesh   = mesh
        self.space  = space
        self.colors = colors
        self.ui     = ui
        self.iscut  = False
    
    def toggleseam( self ):
        self.iscut = not self.iscut

    def draw( self, t, worldspace=True): #slider ):
       #t = ( slider.value - slider.min ) / ( slider.max - slider.min )
       #t = max( t, .06)

        otw = vecutil.vecspace2( self.ui.getpole(), self.ui.getspin() )
        wto = numpy.linalg.inv( otw )

        points = self.mesh.xformresult( wto )

        if self.space == self.kxyz:
            vecutil.xyz2polar( points )

        if self.iscut:
            flatpoints = points.copy()
        
        points[:,1] *= t
        points[:,2] *= t

        vecutil.polar2xyz( points )

        points[:,1] /= t
        points[:,2] /= t

        if worldspace:
            points = vecutil.pointmatrixmult( points, otw )

        if self.iscut:
            edgelengths = self.mesh.edgelengths(flatpoints)
       
            longedges =  edgelengths < math.pi * .5

            erase = self.mesh.binarysignal_e2f( longedges, numpy.minimum )

            idxs = self.mesh.indicies[erase]
            points = points[idxs]
            vptr = vecutil.numpy2pointer(points)
            cptr = vecutil.numpy2pointer(self.colors[idxs])
        
            n = len(points)*3

            idxs = numpy.cast[constants.INTDTYPE](numpy.mgrid[0:n] )

            iptr = vecutil.numpy2pointer(idxs)
        else:
            idxs = self.mesh.indicies
            vptr = vecutil.numpy2pointer(points)
            cptr = vecutil.numpy2pointer(self.colors)
        
            n = len(idxs)*3

            iptr = vecutil.numpy2pointer(idxs)

        self.drawVNC( vptr, vptr, cptr, iptr, n )

    def drawVNC( self, vptr, nptr, cptr, iptr, n ):
        gl.glEnable( GL_NORMALIZE )

        gl.glEnableClientState(GL_VERTEX_ARRAY)
        gl.glEnableClientState(GL_NORMAL_ARRAY)
        gl.glEnableClientState(GL_COLOR_ARRAY)
        gl.glVertexPointer(3, GL_FLOAT, 0, vptr)
        gl.glNormalPointer( GL_FLOAT, 0, nptr)
        gl.glColorPointer(3, GL_FLOAT, 0, cptr)
        gl.glDrawElements(self.ui.fillModes[self.ui.fillMode], n, GL_UNSIGNED_INT, iptr)

        gl.glDisableClientState(GL_VERTEX_ARRAY)
        gl.glDisableClientState(GL_NORMAL_ARRAY)
        gl.glDisableClientState(GL_COLOR_ARRAY)

class ViewerWindow(pyglet.window.Window):
    PADDING = 4
    BUTTON_HEIGHT = 16
    BUTTON_WIDTH  = 45
    _width = 640
    _height = 480

    kpan       = 'kpan'
    kzoom      = 'kzoom'
    korbit     = 'korbit'
    kpolemanip = 'kpolemanip'
    kspinmanip = 'kspinmanip'

    def __init__(self, player):
        config = Config(sample_buffers=1, samples=4,
                    depth_size=24, double_buffer=True,
                    red_size=8, blue_size=8, green_size=8, alpha_size=8 )
        try:
            pyglet.window.Window.__init__(self, caption='seam',
                                               visible=False, 
                                               resizable=True,
                                               config=config)
        except:
            pyglet.window.Window.__init__(self, caption='seam',
                                               visible=False, 
                                               resizable=True)

        self.solverKindButton = controls.TextButton(self)
        self.solverKindButton.on_press = self.togglesolverkind

        self.squishScaleSlider = controls.Slider(self)
        self.squishScaleSlider.value = .1
        self.squishScaleSlider.min = 0
        self.squishScaleSlider.max = 1
        self.squishScaleSlider.on_change = self.squishscalescroll

        self.seedSlider = controls.Slider(self)
        self.seedSlider.value = 0
        self.seedSlider.min = 0
        self.seedSlider.max = 10000
        self.seedSlider.on_change = self.seedscroll

        self.unwrapSlider = controls.Slider(self)
        self.unwrapSlider.value = 100
        self.unwrapSlider.min = 0
        self.unwrapSlider.max = 100
        self.unwrapSlider.on_change = self.scroll

        self.fillModeButton = controls.TextButton(self)
        self.fillModeButton.on_press = self.toggleFillMode

        self.showAxisButton = controls.TextButton(self)
        self.showAxisButton.on_press = self.toggleAxis

        self.cutSeamButton = controls.TextButton(self)
        self.cutSeamButton.on_press = self.toggleSeam
        self.cutSeamButton.text = 'cut'

        self.runButton = controls.TextButton(self)
        self.runButton.on_press = self.run
        self.runButton.text = 'run'

        self.controls = [
            self.unwrapSlider 
           ,self.fillModeButton
           ,self.showAxisButton
           ,self.cutSeamButton
           ,self.seedSlider
           ,self.squishScaleSlider
           ,self.runButton
           ,self.solverKindButton
        ]
        
        for i,control in enumerate(self.controls):
            control.width  = self.BUTTON_WIDTH
            control.height = self.BUTTON_HEIGHT
            control.x = self.PADDING
            control.y = self.PADDING * (i+1) + self.BUTTON_HEIGHT * i

        self.manipstate = None

        self.manipstart = None
        self.viewtransform = Transform()
        self.manipviewtransform = None

        self.mousescreen = vecutil.vec3()
        self.mouseworld  = vecutil.vec3()
        self.pole        = vecutil.vec3(0,0,1)
        self.manippole   = None
        self.spin        = vecutil.vec3(-1,0,0)
        self.manipspin   = None

        self.mesh        = None
        self.solver      = None
        self.solverDirty = True

        self.view = Frustum().rtnf(( 1, 1, -5, 5 )).orthographic()
        self.view.ctw = vecutil.mat4()

        self.zup = Transform()
        self.zup._order = ( self.zup.kscale
                           ,self.zup.kxrotate
                           ,self.zup.kyrotate
                           ,self.zup.kzrotate
                           ,self.zup.ktranslate )
        self.zup.ry = math.radians( -90 )
        self.zup.rx = math.radians( -90 )

        self.solverKind = 0
        self.solverKinds = ( SeamSolver.kzup, SeamSolver.kplane, SeamSolver.kfree )
        self.solverKindNames = ( 'zup', 'plane', 'free' )
        self.solverKindButton.text = self.solverKindNames[self.solverKind]

        self.fillMode = 1
        self.fillModes = ( gl.GL_LINES, gl.GL_TRIANGLES, gl.GL_POINTS)
        self.fillModeNames = ( 'lines', 'solid', 'pnts' )
        self.fillModeButton.text = self.fillModeNames[self.fillMode]

        self.showAxis = True
        self.showAxisNames = ('!xyz','xyz')
        self.showAxisButton.text = self.showAxisNames[ self.showAxis ]

        self.setup()
       #self.run() 

    def setup(self):
        if self.solverDirty:
            print('setting up')
            levels       = 5
            nfrustums    = 25
            frustumscale = .4
            squishscale  = self.squishScaleSlider.value
            solver_kind  = self.solverKinds[self.solverKind]
            seed         = self.seedSlider.value

            if not self.solver or ( self.solver.levels != levels or \
                                 self.solver.kind != solver_kind ):
                self.solver = SeamSolver( levels, solver_kind )
        
            squish = numpy.identity(4)

            if solver_kind == SeamSolver.kplane:
                squishvec = vecutil.randunitpt( seed )
                squish = vecutil.vecspace( squishvec  )
            if solver_kind in (SeamSolver.kzup, SeamSolver.kplane):
                squish[2, :] *= squishscale

            def randfrustums( n, scl, solver, squish=None, seedstart=0 ):
                'randomly distributed view frustums'
                mindimension = solver.mesh.edgestats()[2]
                result = []
                for i in xrange(n):
                    f = Frustum.rand( scl, i + seedstart, mindimension, squish)
                    result.append( f )
                return result

            frustums = randfrustums( nfrustums, 
                                     frustumscale,
                                     self.solver, 
                                     squish,
                                     seed )

            self.solver.markfrustums(frustums)

            colors = self.solver.mesh.points.copy()
            colors[:, 0] = 1-self.solver.vertweight
            colors[:, 1] = self.solver.vertvis

            self.mesh = TriMeshGL( self.solver.mesh, colors=colors, ui=self )

            self.solverDirty = False

    def run(self):
        '''computes frustum/mesh intersection and other stuff
        build a pyramid, build frustums and find seams'''
        seams = self.solver.run()

        if seams:
            dummy, vis, f = seams[0]
            self.pole = f.ctw[2,0:3]
            self.spin = -1 * f.ctw[0,0:3]
            
            self.mesh.colors[vis] = 1, 1, 0

    def getpole(self):
        if self.manippole is not None:
            return self.manippole
        return self.pole

    def getspin(self):
        if self.manipspin is not None:
            return self.manipspin
        return self.spin

    def pixel2screen(self, x, y, z=0.0):
        sx = x / float(self._width)
        sy = y / float(self._height)
        sy = 1.0 - sy

        xrange = self.view.right - self.view.left; 
        yrange = self.view.bottom -  self.view.top;
        xmin = self.view.left; 
        ymin = self.view.top; 
        
        cx = (sx*xrange)+xmin;
        cy = (sy*yrange)+ymin;

        return vecutil.vec3(cx,cy,z)

    def pixel2world_yup(self, x, y, z=0.0):
        screenpos = self.pixel2screen( x, y, z)
        return vecutil.pointmatrixmult( screenpos, self.viewtransform.wto )[0]

    def pixel2world_zup(self, x, y, z=0.0):
        screenpos = self.pixel2screen( x, y, z)
        yuppos = vecutil.pointmatrixmult( screenpos, self.viewtransform.wto )[0]
        return   vecutil.pointmatrixmult( yuppos, self.zup.wto )[0]

    def world2screen_zup(self, pt ):
        yup    = vecutil.pointmatrixmult( pt, self.zup.otw  )[0]
        camera = vecutil.pointmatrixmult( yup, self.viewtransform.otw  )[0]
        return   vecutil.pointmatrixmult( camera, self.view.cts  )[0]

    def on_mouse_press(self, x, y, button, modifiers):

        controlfound = False
        for control in self.controls:
            if control.hit_test(x, y):
                controlfound = True
                control.on_mouse_press(x, y, button, modifiers)
        
        if not controlfound:

            self.manipstart    = vecutil.vec3(x,y)
            self.mousescreen   = self.pixel2screen(x, y)
            self.mouseworld    = self.pixel2world_yup( x, y)
            self.manippole     = self.pole.copy()
            self.manipspin     = self.spin.copy()

            polescreen = self.world2screen_zup( self.pole )
            spinscreen = self.world2screen_zup( self.spin )

            aspect = self._height/float(self._width)
            polescreen[1] *= aspect
            spinscreen[1] *= aspect
            polescreen[2] = 0
            spinscreen[2] = 0
           
            orbit = bool( modifiers & key.MOD_ALT )
            zoom  = bool( modifiers & key.MOD_SHIFT )

            poledist  =  vecutil.vlength( polescreen - self.mousescreen )
            pole      =  poledist < .02

            spindist  =  vecutil.vlength( spinscreen - self.mousescreen )
            spin      =  spindist < .02

            states = [ orbit, zoom, pole, spin ]
            states += [ not any(states) ]

            states = zip( states, (self.korbit, self.kzoom, self.kpolemanip, self.kspinmanip, self.kpan) )
            for value, state in states:
                if value:
                    self.manipstate = state
                    return


        else:
            self.manipstart = None


    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            print('space test')
        elif symbol == key.ESCAPE:
            self.dispatch_event('on_close')

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):

        if self.manipstart is None: return

        current = vecutil.vec3(x,y)
        delta = current - self.manipstart

        self.manipviewtransform = Transform(self.viewtransform)

        if self.manipstate == self.korbit:
            self.manipviewtransform.ry += delta[0] *  .01
            self.manipviewtransform.rx += delta[1] * -.01

        elif self.manipstate == self.kzoom:
            current = self.pixel2screen(x, y)
        
            zoom = vecutil.vlength( current ) /  vecutil.vlength( self.mousescreen )
            self.manipviewtransform.sx = max( zoom * self.manipviewtransform.sx, .00001)
            self.manipviewtransform.sy = max( zoom * self.manipviewtransform.sy, .00001)
            self.manipviewtransform.sz = max( zoom * self.manipviewtransform.sz, .00001)
    
        elif self.manipstate == self.kpan:
            delta = self.pixel2world_yup( x, y ) - self.mouseworld
            self.manipviewtransform.tx += delta[0]
            self.manipviewtransform.ty += delta[1]
            self.manipviewtransform.tz += delta[2]

        if self.manipstate in (self.kpolemanip, self.kspinmanip):
            p0 = self.pixel2world_zup( x, y, 10 )
            p1 = self.pixel2world_zup( x, y, -10 )

            hits = vecutil.sphere_line_intersection( p0, p1, vecutil.vec3(), 1.0 )
            hit = hits[-1]
            if hit is None:
                hit = hits[0]
            if hit is not None:
                if self.manipstate == self.kpolemanip:
                    self.manippole = vecutil.vec3( *hit )
            
                elif self.manipstate == self.kspinmanip:
                    self.manipspin = vecutil.vec3( *hit )

    def on_mouse_release(self, x, y, button, modifiers):
        if self.manipviewtransform:
            self.viewtransform = self.manipviewtransform

        if self.manippole is not None:
            self.pole = self.manippole

        if self.manipspin is not None:
            self.spin = self.manipspin

        self.manipviewtransform = None
        self.manippole = None
        self.manipspin = None

        self.setup()

    def scroll( self, value ):
        self.unwrapSlider.value = value

    def seedscroll( self, value ):
        self.seedSlider.value = value
        self.solverDirty = True

    def squishscalescroll( self, value ):
        self.squishScaleSlider.value = value
        self.solverDirty = True

    def togglesolverkind(self):
        self.solverKind = ( self.solverKind + 1 ) % len( self.solverKinds )
        self.solverKindButton.text = self.solverKindNames[self.solverKind]
        self.solverDirty = True

    def toggleFillMode(self):
        self.fillMode = ( self.fillMode + 1 ) % len( self.fillModes ) 
        self.fillModeButton.text = self.fillModeNames[self.fillMode]

    def toggleAxis(self):
        self.showAxis = not self.showAxis
        self.showAxisButton.text = self.showAxisNames[self.showAxis]

    def toggleSeam(self):
        self.mesh.toggleseam()

    def drawlines( self, verts, colors, idxs ):
        vptr = vecutil.numpy2pointer(verts)
        iptr = vecutil.numpy2pointer(idxs)

        if colors is not None:
            cptr = vecutil.numpy2pointer(colors)
            gl.glEnableClientState(GL_COLOR_ARRAY)
            gl.glColorPointer(3, GL_FLOAT, 0, cptr)

        gl.glEnableClientState(GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, GL_FLOAT, 0, vptr)
        gl.glDrawElements(GL_LINES, len(idxs), GL_UNSIGNED_INT, iptr)
        gl.glDisableClientState(GL_VERTEX_ARRAY)
        gl.glDisableClientState(GL_COLOR_ARRAY)

    def draw_axes(self):
        gl.glPushMatrix()
        gl.glScalef( 1.1, 1.1, 1.1)

        o = 0, 0, 0
        x = 1, 0, 0
        y = 0, 1, 0
        z = 0, 0, 1

        verts  = numpy.array([ o, x, o, y, o, z], dtype=constants.DTYPE )
        colors = numpy.array([ x, x, y, y, z, z], dtype=constants.DTYPE )
        idxs   = numpy.cast[constants.INTDTYPE]( numpy.mgrid[:6] )

        self.drawlines( verts, colors, idxs)

        def draw_axis_label( name, xyz):
            gl.glPushMatrix()
            gl.glTranslatef( *xyz )
            gl.glScalef( .01, .01, .01 )
            gl.glRotatef( 90, 0, 1, 0 )
            gl.glRotatef( 90, 0, 0, 1 )
            pyglet.text.Label(name).draw()
            glPopMatrix()

        draw_axis_label( 'x', x)
        draw_axis_label( 'y', y)
        draw_axis_label( 'z', z)
        gl.glPopMatrix()


    def drawseam( self ):
        pole = self.getpole()
        spin = self.getspin()

        def drawpoint( pt ):
            glPointSize( 6 )
            gl.glBegin( GL_POINTS )
            gl.glColor3f( 1, 1, 1)
            gl.glVertex3d( *pt )
            gl.glEnd()

            glPointSize( 4 )
            gl.glBegin( GL_POINTS )
            gl.glColor3f( 0, 0, 0)
            gl.glVertex3d( *pt )
            gl.glEnd()

            glPointSize( 3 )

        drawpoint( pole )       
        drawpoint( spin )       

        n = 60
        idxs    = numpy.cast[constants.INTDTYPE]( numpy.mgrid[:n] )
        hcircle = numpy.cast[constants.DTYPE]( numpy.mgrid[:n] )
        hcircle /= n
        hcircle -= .5
        hcircle *= math.pi

        circlepts = numpy.zeros( (n,3), dtype = constants.DTYPE)
        circlepts[:,0] = -numpy.cos( hcircle )
        circlepts[:,2] =  numpy.sin( hcircle )
        
        gl.glPushMatrix()

        polemx = vecutil.vecspace2( pole, spin )
        glMultMatrix( polemx )

        gl.glColor3f( 1, 1, 0)
        self.drawlines( circlepts, None, idxs )

        gl.glPopMatrix()
        gl.glColor3f( 1, 1, 1)
   
 
    def on_draw(self):
        gl.glClearColor( .2, .2, .2, 0.)
        self.clear()

        aspect = self._height/float(self._width)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        self.view.rtnf(( 1, aspect, -5, 5 )).orthographic()

        gl.glOrtho( self.view.left
                   ,self.view.right
                   ,self.view.bottom
                   ,self.view.top
                   ,self.view.near
                   ,self.view.far)

        gl.glMatrixMode(gl.GL_MODELVIEW )
        gl.glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )

        gl.glEnable(GL_CULL_FACE)

        gl.glPushMatrix()

        xform = self.viewtransform
        if self.manipviewtransform:
            xform = self.manipviewtransform

        gl.glLoadIdentity()
        xform.glMultMatrix()

        self.zup.glMultMatrix()
     
        slider = self.unwrapSlider
        t = ( slider.value - slider.min ) / ( slider.max - slider.min )
        t = max( t, .06)
        self.mesh.draw( t )

        if self.showAxis:
            self.draw_axes()

        self.drawseam()

        gl.glPopMatrix()

        gl.glLoadIdentity()

    
        gl.glPushMatrix()
        gl.glTranslatef( -1, aspect, 0)
        gl.glScalef( .5, .5, .5)
        gl.glTranslatef( 1.0, -.5, 0)
        gl.glScalef( 1.0/math.pi, 1.0/math.pi, 1)
        
        gl.glRotatef( -90, 0, 1, 0 )
        gl.glRotatef( -90, 1, 0, 0 )
        self.mesh.draw( 0.06, False )
        gl.glPopMatrix()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self._width, 0, self._height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
        for control in self.controls:
            control.draw()

        gl.glLoadIdentity()

    def on_resize(self, width, height):
        self._width = width
        self._height = height
        return pyglet.window.Window.on_resize(self, width, height)

if __name__ == '__main__':
    window = ViewerWindow(None)
    window.set_visible(True)

    pyglet.clock.schedule_interval(lambda dt: None, 0.2)

    pyglet.app.run()
