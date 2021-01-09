import os
import sys
import numpy as np
import numpy.matlib
from scipy.interpolate import interp2d, griddata
import glob
import matplotlib.pyplot as plt
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy.matlib
import pycuda.gpuarray
from math import ceil
import time
from GPUFuncs_pycuda import *
import logging

# define logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class volume_constructor(object):
    def __init__(self, params):
        [self.nx, self.ny, self.nz] = params["NumberOfImage"]
        [self.dx, self.dy, self.dz] = params["ImagePixelSpacing"]
        self.Origin = np.array(params["RotationOrigin"])
        self.PhantomCenter = np.array(params["PhantomCenter"])

    def source_from_file(self, filename):
        pass

    def source_calculation(self, angle):
        pass
