from distutils.core import setup
from distutils.extension import Extension
import os
import sys

ext = Extension('imagetask.core', [])
ext.language = 'c++'

if sys.platform == 'darwin':
    ext.extra_compile_args = ['-isysroot', '/Developer/SDKs/MacOSX10.5.sdk']
    ext.extra_link_args    = ext.extra_compile_args
 
for dir, sub_dirs, files in os.walk( 'imagetask' ):
    ext.include_dirs.append(dir)
    for f in files:
        if f.endswith( '.cpp' ):
            ext.sources.append(os.path.join(dir,f) )

ext.include_dirs.append('.')

setup(name='imagetask',
      version='0.1',
      packages=['imagetask'],
      ext_modules=[ext]
      )
