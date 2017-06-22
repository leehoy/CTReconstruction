import numpy as np
import time

class Forward:
	def __init__(self, filename, params):
		self.parms = {'SourceInit':[0, 0, 0], 'DetectorInit':[0, 0, 0], 'StartAngle':0,
			'EndAngle':0, 'NumberOfDetectorPixels':[0, 0], 'DetectorPixelSize':[0, 0],
			'NumberOfViews':0, 'ImagePixelSpacing':[0, 0, 0], 'NumberOfImage':[0, 0, 0],
			'PhantomCenter':[0, 0, 0]}

	def distance(self):
		SourceToDetector = self.parms['SourceToDetector']
		SourceToAxis = self.parms['SourceToAxis']
		NumberOfViews = self.parms['NumberOfViews']
		AngleCoverage = self.parms['AngleCoverage']
		DetectorChannels = self.parms['DetectorChannels']
		DetectorRows = self.parms['DetectorRows']
		DetectorPixelHeight = self.parms['DetectorPixelHeight']
		DetectorPixelWidth = self.parms['DetectorPixelWidth']
		# Calculates detector center
		angle = np.linspace(0, AngleCoverage, NumberOfViews + 1)
		for i in range(NumberOfViews):
			DetectorCenter = []
			CellCenters = []
			# calculates detector center for each view
			# calculates center of each cell
			# caculates boundary coordinates of each cell
			# create line for each boundary to source coordinate
			# difference plane for condition
			for j in range(DetectorChannels):
				for k in range(DetectorRows):
					pass
				
	def ray_siddons(self):
		nViews = self.params['NumberOfViews']
		[nu, nv] = self.params['NumberOfDetectorPixels']
		[dv, du] = self.params['DetectorPixelSize']
		[dx, dy, dz] = self.params['ImagePixelSpacing']
		[nx, ny, nz] = self.params['NumberOfImage']
		Source_Init = self.params['SourceInit']
		Detector_Init = self.params['DetectorInit']
		StartAngle = self.params['StartAngle']
		EndAngle = self.params['EndAngle']
		Origin = self.params['Origin']
		PhantomCenter = self.params['PhantomCenter']
		
		SAD = np.sqrt(np.sum((Source_Init - Origin) ** 2))
		SDD = np.sqrt(np.sum((Source_Init - Detector_Init) ** 2))
		theta = np.linspace(StartAngle, EndAngle, nViews + 1)
		theta = theta[0:-2]
		Xplane = (PhantomCenter[0] - nx / 2 + range(0, nx)) * dx
		Yplane = (PhantomCenter[1] - ny / 2 + range(0, ny)) * dy
		Zplane = (PhantomCenter[2] - nz / 2 + range(0, nz)) * dz
		Xplane = Xplane - dx / 2
		Yplane = Yplane - dy / 2
		Zplane = Zplane - dz / 2
		proj = np.zeros([nViews, nu, nv], dtype=np.float32)
		for angle in theta:
			SourceX = -SAD * sin(angle)
			SourceY = SAD * cos(angle)
			SourceZ = 0
			
		if(save):
			proj.tofile(write_filename, sep='', format='')
		
		return proj
def main():
	start_time = time.time()
	filename = ''
	F = Forward(filename)
	end_time = time.time()
if __name__ == '__main__':
	main()
