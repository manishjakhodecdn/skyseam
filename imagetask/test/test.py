from imagetask import exrinfo
from imagetask import exrinfoext
import os
from pprint import pprint
import logging
logging.getLogger().setLevel(0)

def go( path):
    info = exrinfoext.ExrInfoExt( path, read_buffer=True )

    import numpy
    numpy.set_printoptions( suppress=True )
    pprint(info.__dict__)
    if info.buffer is not None:
        from PyQt4 import QtGui

        info.shuffleChannels([2,1,0,3])

        result = QtGui.QImage(info.buffer.data, info.width, info.height, QtGui.QImage.Format_ARGB32_Premultiplied)

        result.save(path.replace('exr','png'))

import glob
for path in glob.glob(os.path.dirname(os.path.abspath(__file__))+'/*.exr'):
    go(path)
