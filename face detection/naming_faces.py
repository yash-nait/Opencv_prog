import numpy as np 
import os
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def KNN(X_train,Y_train,query,k=13):
    m=[]
    for i in range(X_train.shape[0]):
        d=dist(X_train[i],query)
        m.append((d,int(Y_train[i])))

    m=sorted(m)
    val=m[:k]
    val=np.array(val)
    max_val=np.unique(val[:,1],return_counts=True)
    idx=max_val[1].argmax()
    return max_val[0][idx]


cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_data=[]
face_id=[]
per_id=()
names={}
curr_id=0
for fx in os.listdir(dir_path):
	if fx.endswith('.npy'):
		data=np.load(dir_path+"//"+fx)
		names[curr_id]=fx[:-4]
		corr_id=curr_id*np.ones((data.shape[0],))
		face_data.append(data)
		face_id.append(corr_id)
		curr_id+=1

X_train=np.concatenate(face_data,axis=0)
Y_train=np.concatenate(face_id,axis=0)
print(X_train.shape)
print(Y_train.shape)

while True:
	ret,frame=cap.read()
	if ret==False:
		continue
	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	for (x,y,w,h) in faces:
		offset=10
		face_section=frame[y-offset : y+h+offset,x-offset : x+w+offset]
		face_section=cv2.resize(face_section,(100,100))
	
		pre=KNN(X_train,Y_train,face_section.flatten())

		per=names[pre]
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
		cv2.putText(frame,per,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		
	cv2.imshow('name',frame)

	key_p=cv2.waitKey(1) & 0xFF

	if key_p==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()	