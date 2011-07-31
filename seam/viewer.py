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
'opengl / pyglet based viewer for seam'

import math
import numpy
import pyglet
import vecutil
import controls
import constants
from pyglet import gl
from frustum import Frustum
from solver import SeamSolver
from transform import Transform

def glmultmatrix( mx ):
    'transform the gl stream'
    mxptr = vecutil.numpy2pointer( mx )
    size = mx.dtype.itemsize

    if size not in (4, 8):
        raise Warning('matrix data must be 4 or 8 bytes')
    if size == 8:
        pyglet.gl.glMultMatrixd( mxptr )
    else:
        pyglet.gl.glMultMatrixf( mxptr )

class TriMeshGL( object ):
    'OpenGL wrapper for a TriMesh'
    kxyz = 'kxyz'
    kruv = 'kruv'
    def __init__(self, mesh, space=kxyz, colors=None, ui=None ):
        self.mesh   = mesh
        self.space  = space
        self.colors = colors
        self.ui     = ui
        self.iscut  = False
    
    def toggleseamcut( self ):
        'flip cut mode'
        self.iscut = not self.iscut

    def draw( self, unwrapamount, worldspace=True):
        '''deform mesh from polar to xyz coordinates
        unwrapamount specifies how round a shape to draw 0 is flat
        the unwrapped mesh can be oriented to world space or object space'''

        otw = vecutil.vecspace2( self.ui.getpole(), self.ui.getspin() )
        wto = numpy.linalg.inv( otw )

        points = self.mesh.xformresult( wto )

        if self.space == self.kxyz:
            vecutil.xyz2polar( points )

        if self.iscut:
            flatpoints = points.copy()
        
        points[:, 1] *= unwrapamount
        points[:, 2] *= unwrapamount

        vecutil.polar2xyz( points )

        points[:, 1] /= unwrapamount
        points[:, 2] /= unwrapamount

        if worldspace:
            points = vecutil.pointmatrixmult( points, otw )

        if not self.iscut:
            self.drawnvc( points, points, self.colors, self.mesh.indicies)
        else:
            edgelengths = self.mesh.edgelengths(flatpoints)
       
            longedges =  edgelengths < math.pi * .5

            erase = self.mesh.binarysignal_e2f( longedges, numpy.minimum )

            idxs = self.mesh.indicies[erase]
            points = points[idxs]
            colors = self.colors[idxs]

            n = len(points)*3
            idxs = numpy.cast[constants.INTDTYPE](numpy.mgrid[0:n] )

            self.drawnvc( points, points, colors, idxs)

    def drawnvc( self, normals, points, colors, idxs  ):
        '''draw tri mesh using glDrawElements
        using input normals points colors and indexes'''
        n = 1
        for dim in idxs.shape:
            n *= dim

        iptr = vecutil.numpy2pointer(idxs)
        nptr = vecutil.numpy2pointer(normals)
        vptr = vecutil.numpy2pointer(points)
        cptr = vecutil.numpy2pointer(colors)

        mode = self.ui.fillmodes[self.ui.fillmode]

        gl.glPolygonMode( gl.GL_FRONT, mode )
        
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, vptr)
        gl.glNormalPointer( gl.GL_FLOAT, 0, nptr)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, cptr)
        gl.glDrawElements(gl.gl.GL_TRIANGLES, n, gl.GL_UNSIGNED_INT, iptr)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPolygonMode( gl.GL_FRONT, gl.GL_FILL )

class ViewerWindow(pyglet.window.Window):
    'viewer window handles all ui events and most drawing'
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

    def __init__(self ):
        'init gl context ui elements, calls setup to be a gl mesh to draw'
        config = pyglet.gl.Config(sample_buffers=1,
                                  samples=4,
                                  depth_size=24,
                                  double_buffer=True,
                                  red_size=8,
                                  blue_size=8,
                                  green_size=8,
                                  alpha_size=8 )
        try:
            pyglet.window.Window.__init__(self, caption='seam',
                                               visible=False, 
                                               resizable=True,
                                               config=config)
        except:
            pyglet.window.Window.__init__(self, caption='seam',
                                               visible=False, 
                                               resizable=True)

        self.solverkindbutton = controls.TextButton(self)
        self.solverkindbutton.on_press = self.togglesolverkind

        self.squishscaleslider = controls.Slider(self)
        self.squishscaleslider.value = .1
        self.squishscaleslider.min = 0
        self.squishscaleslider.max = 1
        self.squishscaleslider.on_change = self.squishscalescroll

        self.seedslider = controls.Slider(self)
        self.seedslider.value = 0
        self.seedslider.min = 0
        self.seedslider.max = 10000
        self.seedslider.on_change = self.seedscroll

        self.unwrapslider = controls.Slider(self)
        self.unwrapslider.value = 100
        self.unwrapslider.min = 0
        self.unwrapslider.max = 100
        self.unwrapslider.on_change = self.scroll

        self.fillemodebutton = controls.TextButton(self)
        self.fillemodebutton.on_press = self.togglefillmode

        self.showaxisbutton = controls.TextButton(self)
        self.showaxisbutton.on_press = self.toggleaxis

        self.cutseambutton = controls.TextButton(self)
        self.cutseambutton.on_press = self.toggleseam
        self.cutseambutton.text = 'cut'

        self.runbutton = controls.TextButton(self)
        self.runbutton.on_press = self.run
        self.runbutton.text = 'run'

       #self.fps_display = pyglet.clock.ClockDisplay()

        self.controls = [
            self.unwrapslider 
           ,self.fillemodebutton
           ,self.showaxisbutton
           ,self.cutseambutton
           ,self.seedslider
           ,self.squishscaleslider
           ,self.runbutton
           ,self.solverkindbutton
        ]
        
        for i, control in enumerate(self.controls):
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
        self.pole        = vecutil.vec3(0, 0, 1)
        self.manippole   = None
        self.spin        = vecutil.vec3(-1, 0, 0)
        self.manipspin   = None

        self.mesh        = None
        self.solver      = None
        self.solverdirty = True

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

        self.solverkind = 0
        self.solverkinds = SeamSolver.kzup, SeamSolver.kplane, SeamSolver.kfree
        self.solverkindnames = 'zup', 'plane', 'free'
        self.solverkindbutton.text = self.solverkindnames[self.solverkind]

        self.fillmode = 1
        self.fillmodes = gl.GL_LINE, gl.GL_FILL, gl.GL_POINT
        self.fillmodenames = 'lines', 'solid', 'pnts' 
        self.fillemodebutton.text = self.fillmodenames[self.fillmode]

        self.showaxis = True
        self.showaxisnames = '!xyz','xyz'
        self.showaxisbutton.text = self.showaxisnames[ self.showaxis ]

        self.setup()

    def setup(self):
        '''called from event loop, will regen a solver whenever dirty
        the number and extent of frustums is not currently set from ui
        creates random views distributed in a possibly squished sphere
        sets up the solver to find the best seam for the random views
        '''
        if self.solverdirty:
            levels       = 5
            nfrustums    = 25
            frustumscale = .4
            squishscale  = self.squishscaleslider.value
            solver_kind  = self.solverkinds[self.solverkind]
            seed         = self.seedslider.value

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

            self.solverdirty = False

    def run(self):
        '''computes frustum/mesh intersection and other stuff
        build a pyramid, build frustums and find seams'''
        seams = self.solver.run()

        if seams:
            dummy, vis, f = seams[0]
            self.pole = f.ctw[2, 0:3]
            self.spin = -1 * f.ctw[0, 0:3]
            
            self.mesh.colors[vis] = 1, 1, 0

    def getpole(self):
        'gets current pole vector: may be during a mouse drag'
        if self.manippole is not None:
            return self.manippole
        return self.pole

    def getspin(self):
        'gets current spin vector: may be during a mouse drag'
        if self.manipspin is not None:
            return self.manipspin
        return self.spin

    def pixel2screen(self, x, y, z=0.0):
        'transforms pixel coordinates to normalized device coords'
        sx = x / float(self._width)
        sy = y / float(self._height)
        sy = 1.0 - sy

        rangex = self.view.right - self.view.left
        rangey = self.view.bottom -  self.view.top
        xmin = self.view.left
        ymin = self.view.top
        
        cx = (sx*rangex)+xmin
        cy = (sy*rangey)+ymin

        return vecutil.vec3(cx, cy, z)

    def pixel2world_yup(self, x, y, z=0.0):
        'pixel projected to world space without swapping y/z'
        screenpos = self.pixel2screen( x, y, z)
        return vecutil.pointmatrixmult( screenpos, self.viewtransform.wto )[0]

    def pixel2world_zup(self, x, y, z=0.0):
        'pixel projected to world space with swapping y/z'
        screenpos = self.pixel2screen( x, y, z)
        yuppos = vecutil.pointmatrixmult( screenpos, self.viewtransform.wto )[0]
        return   vecutil.pointmatrixmult( yuppos, self.zup.wto )[0]

    def world2screen_zup(self, pt ):
        'transform world coordinate to screen space'
        yup    = vecutil.pointmatrixmult( pt, self.zup.otw  )[0]
        camera = vecutil.pointmatrixmult( yup, self.viewtransform.otw  )[0]
        return   vecutil.pointmatrixmult( camera, self.view.cts  )[0]

    def on_mouse_press(self, x, y, button, modifiers):
        'hit test controls, possibly start direct manip'

        controlfound = False
        for control in self.controls:
            if control.hit_test(x, y):
                controlfound = True
                control.on_mouse_press(x, y, button, modifiers)
        
        if not controlfound:

            self.manipstart    = vecutil.vec3(x, y)
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
           
            orbit = bool( modifiers & pyglet.window.key.MOD_ALT )
            zoom  = bool( modifiers & pyglet.window.key.MOD_SHIFT )

            poledist  =  vecutil.vlength( polescreen - self.mousescreen )
            pole      =  poledist < .02

            spindist  =  vecutil.vlength( spinscreen - self.mousescreen )
            spin      =  spindist < .02

            states = [ orbit, zoom, pole, spin ]
            states += [ not any(states) ]

            states = zip( states, (self.korbit
                                  ,self.kzoom
                                  ,self.kpolemanip
                                  ,self.kspinmanip
                                  ,self.kpan ))
            for value, state in states:
                if value:
                    self.manipstate = state
                    return
        else:
            self.manipstart = None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        'orbit, zoom or pan view'

        if self.manipstart is None:
            return

        current = vecutil.vec3(x, y)
        delta = current - self.manipstart

        self.manipviewtransform = Transform(self.viewtransform)

        if self.manipstate == self.korbit:
            self.manipviewtransform.ry += delta[0] *  .01
            self.manipviewtransform.rx += delta[1] * -.01

        elif self.manipstate == self.kzoom:
            current = self.pixel2screen(x, y)
        
            zoom = vecutil.vlength(current) / vecutil.vlength(self.mousescreen)
           
            eps =  .00001
            self.manipviewtransform.sx *= zoom
            self.manipviewtransform.sy *= zoom
            self.manipviewtransform.sz *= zoom
            self.manipviewtransform.sx = max( self.manipviewtransform.sx, eps)
            self.manipviewtransform.sy = max( self.manipviewtransform.sy, eps)
            self.manipviewtransform.sz = max( self.manipviewtransform.sz, eps)
    
        elif self.manipstate == self.kpan:
            delta = self.pixel2world_yup( x, y ) - self.mouseworld
            self.manipviewtransform.tx += delta[0]
            self.manipviewtransform.ty += delta[1]
            self.manipviewtransform.tz += delta[2]

        if self.manipstate in (self.kpolemanip, self.kspinmanip):
            p0 = self.pixel2world_zup( x, y, 10 )
            p1 = self.pixel2world_zup( x, y, -10 )

            hits = vecutil.sphere_line_intersection( p0, 
                                                     p1, 
                                                     vecutil.vec3(), 
                                                     1.0 )
            hit = hits[-1]
            if hit is None:
                hit = hits[0]
            if hit is not None:
                if self.manipstate == self.kpolemanip:
                    self.manippole = vecutil.vec3( hit[0], hit[1], hit[2] )
            
                elif self.manipstate == self.kspinmanip:
                    self.manipspin = vecutil.vec3( hit[0], hit[1], hit[2] )

    def on_mouse_release(self, x, y, button, modifiers):
        'commits any manipulations and runs setup to update solver'
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

    def on_key_press(self, symbol, modifiers):
        'escape closes window -- any hotkeys would go here'
        if symbol == pyglet.window.key.SPACE:
            pass
        elif symbol == pyglet.window.key.ESCAPE:
            self.dispatch_event('on_close')

    def scroll( self, value ):
        'unwrap slider callback'
        self.unwrapslider.value = value

    def seedscroll( self, value ):
        'seed slider callback'
        self.seedslider.value = value
        self.solverdirty = True

    def squishscalescroll( self, value ):
        'squish slider callback'
        self.squishscaleslider.value = value
        self.solverdirty = True

    def togglesolverkind(self):
        'toggle through zup plane and free solver modes'
        self.solverkind = ( self.solverkind + 1 ) % len( self.solverkinds )
        self.solverkindbutton.text = self.solverkindnames[self.solverkind]
        self.solverdirty = True

    def togglefillmode(self):
        'toggle through wireframe point and solid poly fill modes'
        self.fillmode = ( self.fillmode + 1 ) % len( self.fillmodes ) 
        self.fillemodebutton.text = self.fillmodenames[self.fillmode]

    def toggleaxis(self):
        'show x y z axis in view'
        self.showaxis = not self.showaxis
        self.showaxisbutton.text = self.showaxisnames[self.showaxis]

    def toggleseam(self):
        'toggle seam cutting for each mesh draw'
        self.mesh.toggleseamcut()

    def drawlines( self, verts, colors, idxs ):
        'helper to draw lines from numpy arrays of verts/colors/indexes'
        vptr = vecutil.numpy2pointer(verts)
        iptr = vecutil.numpy2pointer(idxs)

        if colors is not None:
            cptr = vecutil.numpy2pointer(colors)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, cptr)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, vptr)
        gl.glDrawElements(gl.GL_LINES, len(idxs), gl.GL_UNSIGNED_INT, iptr)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

    def draw_axes(self):
        'draw x y z axis in r g b with text labels'
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
            'draw a single label'
            gl.glPushMatrix()
            gl.glTranslatef( xyz[0], xyz[1], xyz[2] )
            gl.glScalef( .01, .01, .01 )
            gl.glRotatef( 90, 0, 1, 0 )
            gl.glRotatef( 90, 0, 0, 1 )
            pyglet.text.Label(name).draw()
            gl.glPopMatrix()

        draw_axis_label( 'x', x)
        draw_axis_label( 'y', y)
        draw_axis_label( 'z', z)
        gl.glPopMatrix()


    def drawseam( self ):
        'draw manip and dotted line for the seam'
        pole = self.getpole()
        spin = self.getspin()

        def drawpoint( pt ):
            'draw a single point with white outline and black interior'
            gl.glPointSize( 6 )
            gl.glBegin( gl.GL_POINTS )
            gl.glColor3f( 1, 1, 1)
            gl.glVertex3d( pt[0], pt[1], pt[2] )
            gl.glEnd()

            gl.glPointSize( 4 )
            gl.glBegin( gl.GL_POINTS )
            gl.glColor3f( 0, 0, 0)
            gl.glVertex3d( pt[0], pt[1], pt[2] )
            gl.glEnd()

            gl.glPointSize( 3 )

        drawpoint( pole )       
        drawpoint( spin )       

        n = 60
        idxs    = numpy.cast[constants.INTDTYPE]( numpy.mgrid[:n] )
        hcircle = numpy.cast[constants.DTYPE]( numpy.mgrid[:n] )
        hcircle /= n
        hcircle -= .5
        hcircle *= math.pi

        circlepts = numpy.zeros( (n, 3), dtype = constants.DTYPE)
        circlepts[:, 0] = -numpy.cos( hcircle )
        circlepts[:, 2] =  numpy.sin( hcircle )
        
        gl.glPushMatrix()

        polemx = vecutil.vecspace2( pole, spin )
        glmultmatrix( polemx )

        gl.glColor3f( 1, 1, 0)
        self.drawlines( circlepts, None, idxs )

        gl.glPopMatrix()
        gl.glColor3f( 1, 1, 1)
   
 
    def on_draw(self):
        'draw all controls, manips and meshes'
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
        gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST )

        gl.glEnable(gl.GL_CULL_FACE)

        gl.glPushMatrix()

        xform = self.viewtransform
        if self.manipviewtransform:
            xform = self.manipviewtransform

        gl.glLoadIdentity()
        xform.glmultmatrix()

        self.zup.glmultmatrix()
     
        slider = self.unwrapslider
        t = ( slider.value - slider.min ) / ( slider.max - slider.min )
        t = max( t, .06)
        self.mesh.draw( t )

        if self.showaxis:
            self.draw_axes()

        self.drawseam()

        gl.glPopMatrix()

        gl.glLoadIdentity()
    
        gl.glPushMatrix()
        gl.glTranslatef( -1, aspect, 0)
        gl.glScalef( .5, .5, .5)
        gl.glTranslatef( 1.0, -.5, 0)
        gl.glScalef( 1.0/math.pi, 1.0/math.pi, 1)
        
        self.zup.glmultmatrix()
        self.mesh.draw( 0.06, False )
        gl.glPopMatrix()

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self._width, 0, self._height, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        
        for control in self.controls:
            control.draw()

       #gl.glTranslatef( self._width - 200, self._height - 50, 0 )
       #self.fps_display.draw()

        gl.glLoadIdentity()

    def on_resize(self, width, height):
        'keep track of width and hight, let pyglet do the rest of the work'
        self._width = width
        self._height = height
        return pyglet.window.Window.on_resize(self, width, height)

if __name__ == '__main__':
    WINDOW = ViewerWindow()
    WINDOW.set_visible(True)

    pyglet.app.run()
