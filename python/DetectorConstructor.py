import os
import sys
import numpy as np
import numpy.matlib
from scipy.interpolate import interp2d, griddata
import glob
import matplotlib.pyplot as plt
import numpy.matlib
from math import ceil
import time
import logging

# define logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class detector_constructor(object):
    def __init__(self, params):
        self.detector = np.array(params["DetectorInit"], dtype=np.float32)
        self.origin = np.array(params["Origin"], dtype=np.float32)
        self.z0 = self.detector[2]
        [self.du, self.dv] = params["DetectorPixelSize"]
        [self.nu, self.nv] = params["NumberOfDetectorPixels"]
        self.did = np.sqrt(np.sum((self.detector - self.origin) ** 2))
        self.detector_offset = np.array(params["DetectorOffset"], dtype=np.float32)
        self.shape = params["DetectorShape"]
        self.pitch = params["Pitch"]

    def detector_from_file(self, filename):
        pass

    def detector_calculation(self, angle):
        pass