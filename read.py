## Reads the output of the yolo detections and creates a csv table which is
## easier to read later.

import os
import csv

##Reads the output of the yolo detections
with open('result.txt') as f:
    lines = f.readlines()
largo=len(lines)

tabla=[]

## Creates the table
for i in range(0,largo):
	if((lines[i][0:5]=="Enter")&((i+1)<largo)):
		archivo=(lines[i].split(":")[1].split("/")[2].split(".")[0])
		x= int(0) 
		y= int(0) 
		if(lines[i+1][0:4]=="frog"):
			info=lines[i+1].split(":")
			x=int(int(info[2].split("t")[0])+int(info[4].split("h")[0])/2)
			y=int(int(info[3].split("w")[0])+int(info[5].split(")")[0])/2)
		tabla.append([archivo,x,y])
	
## Writes table to csv file
myfile = open('output.csv','w')
wr = csv.writer(myfile, delimiter=',')
wr.writerows(tabla)
myfile.close()