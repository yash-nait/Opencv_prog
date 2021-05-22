import numpy as np
import cv2

cap=cv2.VideoCapture(0)

#Cascade Classifier to detect eyes
eyes_cas=cv2.CascadeClassifier("frontalEyes35x16.xml")

while True:
	ret,frame=cap.read()

	if ret==False:
		continue

	gray_sc=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	eyes = eyes_cas.detectMultiScale(gray_sc,1.3,5)

	for (x,y,w,h) in  eyes:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		# get image for glasses
		g_img=cv2.imread("glasses.png",-1)

		g_img=cv2.resize(g_img,(w,h))

		# overlap glasses on image
		for i in range(g_img.shape[0]):
			for j in range(g_img.shape[1]):
				if g_img[i,j,3]>0 :
					frame[y+i,x+j,:]=g_img[i,j,:-1]


	cv2.imshow("face",frame)

	key_p=cv2.waitKey(1) & 0xff
	
	if key_p==ord('q'):
		break

cap.release()
cap.destroyAllWindows()	