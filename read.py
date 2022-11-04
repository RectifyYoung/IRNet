import cPickle
import os
import json
import pylab
import numpy
from PIL import Image

read_file=open('/root/workspace/server/IP-Net/SS_HOI.pkl','rb')
img=cPickle.load(read_file)
read_file.close()
pylab.imshow(img)
pylab.gray()
pylab.show()

