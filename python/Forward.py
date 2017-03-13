import numpy as np
import time

class Forward:
	def __init__(self,filename):
		self.parms={'SourceToDetector':0,'SourceToAxis':0,'AngleCoverage':0,
			'DetectorChannels':0,'DetectorRows':0,

	def distance(self):
		SourceToDetector=self.parms['SourceToDetector']
		SourceToAxis=self.parms['SourceToAxis']
		NumberOfViews=self.parms['NumberOfViews']
		AngleCoverage=self.parms['AngleCoverage']
		DetectorChannels=self.parms['DetectorChannels']
		DetectorRows=self.parms['DetectorRows']
		DetectorPixelHeight=self.parms['DetectorPixelHeight']
		DetectorPixelWidth=self.parms['DetectorPixelWidth']
		
		angle=np.linspace(0,AngleCoverage,NumberOfViews+1)
		for i in range(NumberOfViews):
			for j in range(DetectorChannels):
				for k in range(DetectorRows):
					CellCenter=[
			
def main():
	start_time=time.time()
	filename=''
	F=Forward(filename)
	end_time=time.time()
if __name__=='__main__':
	main()
