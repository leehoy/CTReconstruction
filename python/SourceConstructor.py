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


class source_constructor(object):
    def __init__(self, params):
        self.source = np.array(params["SourceInit"])
        self.origin = np.array(params["RotationOrigin"])
        self.z0 = self.source[2]
        self.sid = np.sqrt(np.sum((self.source - self.origin) ** 2))
        self.sdd = np.sqrt(
            np.sum(
                (self.source - np.array(params["DetectorInit"], dtype=np.float32)) ** 2
            )
        )
        self.pitch = params["Pitch"]
        self.translatoin_per_rotation = self.pitch * (nv * dv) * self.sid / self.sdd

    def source_from_file(self, filename):
        pass

    def source_calculation(self, angle):
        pass
