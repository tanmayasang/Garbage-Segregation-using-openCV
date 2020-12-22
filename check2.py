import cv2, sys, numpy as np
import csv
import math
import ssl
import urllib.request
import cv2
import webbrowser
import numpy as np
url = 'http://192.168.43.1:8080/shot.jpg'
from statistics import mode, StatisticsError

# read command-line arguments
#filename = sys.argv[1]
#t = int(sys.argv[2])

# read original image
while True:
	#print("shakti")
	context = ssl._create_unverified_context()
	res = urllib.request.urlopen(url,context = context)
	imgNp = np.array(bytearray(res.read()) , dtype = np.uint8)
	font = cv2.FONT_HERSHEY_SIMPLEX
	img = cv2.imdecode(imgNp,-1)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#img = cv2.imread('D:/StudyMaterial/6th sem/Tarp/output/banana.jpg') 

	# create binary image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	#(t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
	(t, binary) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	# find contours
	(_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
	    cv2.CHAIN_APPROX_SIMPLE)

	# draw contours over original image
	cv2.drawContours(binary, contours, -1, (255, 255, 255), -1) #(0,0,255) are the colors and last argument defines the thickness of the contour


	#Elimination small contours
	Areacontours = list()
	for i in Areacontours:
	    area = cv2.contourArea(contours[i])
	    if (area > 90 ):
	        Areacontours.append(contours[i])
	    contours = Areacontours

	# #print('found objects')
	# print(len(contours))

	'''print("humoments")
	mom = cv2.moments(contours[0])
	Humoments = cv2.HuMoments(mom)
	Humoments2 = -np.sign(Humoments)*np.log10(np.abs(Humoments))
	print(Humoments2)'''


	hum=cv2.HuMoments(cv2.moments(binary)).flatten()
	output = ' '.join(str(i) for i in hum)
	#print(output)


	# display original image with contours
	#cv2.namedWindow("output", cv2.WINDOW_NORMAL)
	#cv2.imshow("output", binary)
	#cv2.waitKey(0)


	#################################
	# knn classification

	data = []
	with open('D:/StudyMaterial/6th sem/Tarp/output/output1.csv', newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			temp = []
			for i in range(7):
				temp.append(float(row[i]))
			temp.append(row[7])
			data.append(temp)

	#input number of nearest neighbours
	k = 3

	#calculate distances
	compare = output.split()
	for i in range(7):
		compare[i] = float(compare[i])

	distances = []
	for i in range(len(data)):
		sum_of_sq = 0
		for j in range(7):
			sum_of_sq = sum_of_sq + math.pow((compare[j] - data[i][j]), 2)
		distances.append([math.pow(sum_of_sq, 2), data[i][7]]) 

	#sorting distances in ascending order
	final = sorted(distances, key = lambda x:x[0])

	#taking maximum instances
	classify = []
	for i in range(k):
		classify.append(final[i][1])

	#print('k Nearest Neighbours: ', classify)

	try:
	    res = mode(classify)
	except StatisticsError:
	    res = final[0][1]
	x=50
	y=50
	h=10
	w=10
	cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),2)
	cv2.putText(img,res,(x,y-10),font,0.55,(0,255,0),1)
	print("Classified as ", res)
	cv2.imshow('im',img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



