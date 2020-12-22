import cv2, sys, numpy as np
import csv
import math
import statistics

from os import listdir
from os.path import isfile, join

# read command-line arguments
#filename = sys.argv[1]
#t = int(sys.argv[2])

# read original image
mypath = 'D:/StudyMaterial/6th sem/Tarp/dataset/mobile phones - Google Search/'

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in files:
	img = cv2.imread(mypath + f) 

	# create binary image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	#(t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
	(t, binary) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	# find contours
	(_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
	    cv2.CHAIN_APPROX_SIMPLE)

	# draw contours over original image
	cv2.drawContours(binary, contours, -1, (255, 255, 255), -1) #(0,0,255) are the colors and last argument defines the thickness of contour


	#Elimination small contours
	Areacontours = list()
	for i in Areacontours:
		area = cv2.contourArea(contours[i])
		if (area > 90 ):
			Areacontours.append(contours[i])
		contours = Areacontours

	print('found objects')
	print(len(contours))

	'''print("humoments")
	mom = cv2.moments(contours[0])
	Humoments = cv2.HuMoments(mom)
	Humoments2 = -np.sign(Humoments)*np.log10(np.abs(Humoments))
	print(Humoments2)'''


	hum=cv2.HuMoments(cv2.moments(binary)).flatten()
	output = ' '.join(str(i) for i in hum)

	compare = output.split()

	with open('D:/StudyMaterial/6th sem/Tarp/output/output1.csv', 'a', newline='') as csvfile:
		append = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		append.writerow(compare + ['Mobile phones -> non-biodegradable'])


