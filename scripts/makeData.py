import sys
import numpy as np
import re
import pysam
import pandas as pd
from matplotlib import pyplot

def makeXY (inSam,chr,start,stop):
	temp = np.empty((0,2),int)
	for pileupColumn in inSam.pileup(chr, start, stop):
		temp = np.vstack((temp,[pileupColumn.pos,pileupColumn.n]))
	temp = pd.DataFrame(data=temp[0:,1:],  index=temp[0:,0],columns = ["count"])
	Y_full = pd.DataFrame(data=list(range(start,stop)),columns = ["positions"])
	Y_full['counts'] = Y_full['positions'].map(temp['count'])
	Y_full = Y_full.fillna(0)
	Y = (Y_full['counts']/inSam.mapped)*1000000 #normalize by depth since we are plotting both IP and IN in same scale. 
	return (range(len(Y)),Y)

def makeData (IP,IN,regions,outputDir):
	IP_sam = pysam.AlignmentFile(IP,"rb")
	IN_sam = pysam.AlignmentFile(IN,"rb")
	reg_count =1
	for index,i in regions.iterrows():
		ipXY = makeXY(IP_sam,i[0],int(i[1]),int(i[2]))
		inXY = makeXY(IN_sam,i[0],int(i[1]),int(i[2]))
		pyplot.figure(1,figsize =(2.5,2.5),dpi=96)
		pyplot.plot(ipXY[0],ipXY[1])
		pyplot.plot(inXY[0],inXY[1])
		pyplot.axis('off')
		pyplot.savefig(outputDir+"/"+str(reg_count)+".png")
		pyplot.close()
		reg_count = reg_count+1
	

regions = pd.read_csv(sys.argv[1],sep="\t",header=None)
makeData ("SRR2961602_H1_TKO_ESC_H3K4me3_IP_R2_1_trimmed_sorted.bam","SRR2961598_H1_TKO_ESC_H3K4me1_H3K4me3_input_R2_1_trimmed_sorted.bam",regions,sys.argv[2])