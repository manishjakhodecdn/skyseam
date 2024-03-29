import numpy
import ctypes
import logging

class ExrInfo( object ):
    channels = 'channels'
    Error_p = ctypes.POINTER( ctypes.c_char_p )
    Int_p = ctypes.POINTER( ctypes.c_int )
    Missing_Type_Function_p = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)

    Alloc_Image_Buffer_Function_p = ctypes.CFUNCTYPE(*([ctypes.c_void_p] + [ctypes.c_int]*2 + [Int_p]*2 ))
    Named_Int_Function_p = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    Named_Float_Function_p = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_float)
    Named_Str_Function_p = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
    Named_M44f_Function_p = ctypes.CFUNCTYPE(*( [ctypes.c_int, ctypes.c_char_p] + [ctypes.c_float]*16))
    Named_Box2i_Function_p = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    Named_Channel_Function_p = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
    argtypes = [Error_p, 
                ctypes.c_char_p, 
                ctypes.c_bool,
                Missing_Type_Function_p,
                Alloc_Image_Buffer_Function_p,
                Named_Int_Function_p,
                Named_Float_Function_p,
                Named_Str_Function_p,
                Named_M44f_Function_p,
                Named_Box2i_Function_p,
                Named_Channel_Function_p]

    read_exr_info = None

    @classmethod
    def load_library( cls, libpath=None ):
        if cls.read_exr_info is None:
            if libpath is None:
                import os
                libpath = os.path.join( os.path.dirname(__file__ ), 'core.so' )
            l = numpy.ctypeslib.load_library( libpath, '.' )
            cls.read_exr_info = l.read_exr_info
            cls.read_exr_info.argtypes = cls.argtypes

    def __init__( self, path, read_buffer=True, aspect=None, max_width=256, max_height=256 ):
        self.attributes  = {}
        self.path        = path
        self.aspect      = aspect
        self.max_width   = max_width
        self.max_height  = max_height
        self.width       = None
        self.height      = None
        self.orig_width  = None
        self.orig_height = None
        self.buffer      = None
        self.error       = ''
        self.warnings    = []

        
        error = ctypes.c_char_p()
        error_p = ctypes.pointer( error )

        status = self.read_exr_info(error_p, 
                                    self.path, 
                                    read_buffer,
                                    self.missingType(),
                                    self.allocBuffer(),
                                    self.namedInt(),
                                    self.namedFloat(),
                                    self.namedStr(),
                                    self.namedM44f(),
                                    self.namedBox2i(),
                                    self.namedChannel()
                                    )
        if status != 0:
            self.error = str( error_p[0] )

    def shuffleChannels( self, new_order = None):
        if new_order is None:
            new_order = [2,1,0,3]

        rgba = [self.buffer[:,:,0].copy()
               ,self.buffer[:,:,1].copy()
               ,self.buffer[:,:,2].copy()
               ,self.buffer[:,:,3].copy()]

        for count, newidx in enumerate(new_order):
            self.buffer[:,:,count] = rgba[newidx]

        self.buffer = numpy.ascontiguousarray(self.buffer)

    def sizeFromOrig( self ):
        self.width  = min( self.max_width, self.orig_width )

        if self.aspect is not None:
            aspect = self.aspect
        else:               
            aspect = self.orig_width / float(self.orig_height)

        self.height = int(round( self.width / aspect ))
            
    def allocBuffer( self ):
        def closure( orig_width, orig_height, new_width, new_height):
            self.orig_width  = orig_width
            self.orig_height = orig_height
            
            self.sizeFromOrig()

            new_width[0]  = self.width
            new_height[0] = self.height
            self.buffer   = numpy.zeros( (self.height, self.width, 4), dtype='uint8' )
            
            return self.buffer.ctypes.data
        return self.Alloc_Image_Buffer_Function_p(closure)
            
    def namedInt( self ):
        def closure( name, value ):
            self.attributes[name] = int(value)
            return 0
        return self.Named_Int_Function_p(closure)
            
    def namedFloat( self ):
        def closure( name, value ):
            self.attributes[name] = float(value)
            return 0
        return self.Named_Float_Function_p(closure)
        
    def namedStr( self ):
        def closure( name, value ):
            self.attributes[name] = str(value)
            return 0
        return self.Named_Str_Function_p(closure)

    def namedM44f( self ):
        def closure( *args ):
            name = args[0]
            value = args[1:]
            r = []
            for i in range(16):
                r.append( float(value[i]) )
            self.attributes[name] = numpy.array( r ).reshape((4,4))
            return 0
        return self.Named_M44f_Function_p(closure)

    def namedBox2i( self ):
        def closure( name, minx, miny, maxx, maxy ):
            r = [[int(minx), int(miny)], [int(maxx), int(maxy)]]
            self.attributes[name] = r
            return 0
        return self.Named_Box2i_Function_p(closure)

    def namedChannel( self ):
        def closure( name, value ):
            channels = self.attributes.get( self.channels, [])
            channels.append( (name, value) )
            self.attributes[self.channels] = channels
            return 0
        return self.Named_Channel_Function_p(closure)
    
    def missingType( self ):
        def closure( name, value ):
            self.warnings.append( 'exrinfo:ignoring exr attr[%s] of unknown type[%s]'%(name, value) )
            return 0
        return self.Missing_Type_Function_p(closure)

ExrInfo.load_library()
