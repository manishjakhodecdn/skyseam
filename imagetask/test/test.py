import exrinfo
import exrinfoext
import os
from pprint import pprint
import logging
logging.getLogger().setLevel(0)

path = os.path.dirname(os.path.abspath(__file__))+'/xpos.exr'

info = exrinfoext.ExrInfoExt( path, read_buffer=True )
pprint(info.attributes)

import numpy
numpy.set_printoptions( suppress=True )
print( 'worldToNDC')
print( info.attributes['worldToNDC'] )
if info.buffer is not None:
    from PyQt4 import QtGui
    result = QtGui.QImage(info.buffer.data, info.height, info.width, QtGui.QImage.Format_RGB32)
    result.save('result.png')
