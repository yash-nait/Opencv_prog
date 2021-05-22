import cv2
import numpy as np
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

cap=cv2.VideoCapture(0)

#Cascade Classifier to detect eyes
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

turn=0

face_data=list()

name=input("enter your name : ")

while True:

	ret,frame=cap.read()

	if ret==False:
		continue

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	faces = sorted(faces,key=lambda a:a[2]*a[3])
	
	for (x,y,w,h) in  faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		#extract
		offset=10
		turn+=1

		face_section=frame[y-offset : y+h+offset,x-offset : x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		if turn%10==0:
			face_data.append(face_section)
			print("appended")

		cv2.imshow("AOI",face_section)

	cv2.imshow("face",frame)
	
	key_p=cv2.waitKey(1) & 0xff
	if key_p==ord('q'):
		break

face_data=np.asarray(face_data)

face_data=face_data.reshape((face_data.shape[0],-1))

print(face_data.shape)

np.save(dir_path+"/"+name+".npy",face_data)

cap.release()

cv2.destroyAllWindows()	