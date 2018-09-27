
import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
import playsound
import face_recognition
import glob
import os
import natsort
import datetime
import time

def main():
	known_face_encoding=[]
	path="/home/vinit/Desktop/ML/Self notes/Computer Vision/facial_rec/Pics"
	cap=cv2.VideoCapture(0)

	#Returns list of faces from the captured images
	img_file=os.listdir(path)
	#sorting the image label according to their numerical value written 
	img_file=natsort.natsorted(img_file)  

	for img in img_file:
	        #loads an image(.jpg,.png etc) into numpy array
	        image=face_recognition.load_image_file(path + '/' + img)
	        #128 measurements will be stored in encodings taking 68 landmarks on face for each person
	        encoding=face_recognition.face_encodings(image,num_jitters=1)[0]
	        #num_jitters:How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower)
	        # i.e. 50 = 50 x slower
	        known_face_encoding.append(encoding)
	                                   
	#Returns list of known face names from the text file
	a=open("/home/vinit/Desktop/ML/Self notes/Computer Vision/facial_rec/Names.txt","r").read()
	known_face_names=a.split()

	face_locations=[]
	face_encodings=[]
	face_names=[]

	def ratio(eye):
		x=distance.euclidean(eye[1],eye[5])
		y=distance.euclidean(eye[2],eye[4])
		z=distance.euclidean(eye[0],eye[3])

		R=(x+y)/(2.0*z)
		return R

	def play(path):
		playsound.playsound(path)

	#ratio must fall below threshold and then rise for a blink
	threshold_ratio=0.20
	#2 conse.frames must be there with ratio less than threshold for a blink
	#Why?- as it may happen for a frame,ratio goes down and then comes up quickly
	consecutive_frames=25

	#counter for the frame
	fr_counter=0
	Alarm_on=False

	detector=dlib.get_frontal_face_detector()
	predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	#grabbing index of eyes
	(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


	process_this_frame=True

	while True:
		ret,frame=cap.read()
		small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
		#converting BGR to RGB format as dlib supports RGB
		rgb=small_frame[:,:,::-1]
		
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces=detector(gray,0)
		for i in faces:
			embedding=predictor(gray,i)
			embedding=face_utils.shape_to_np(embedding)

			left_eye=embedding[lStart:lEnd]
			right_eye=embedding[rStart:rEnd]

			lratio=ratio(left_eye)
			rratio=ratio(right_eye)

			avg_ratio=(lratio+rratio)/2.0

			#making circles around the eyes
			lhull=cv2.convexHull(left_eye)
			rhull=cv2.convexHull(right_eye)

			cv2.drawContours(frame,[lhull],-1,(0,0,0),1)
			cv2.drawContours(frame,[rhull],-1,(0,0,0),1)

			if avg_ratio<threshold_ratio:
				fr_counter=fr_counter+1

				if fr_counter>=consecutive_frames:
					if not Alarm_on:
						Alarm_on=True
						play("alarm2.mp3")
					cv2.putText(frame,"Don't Sleep!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

			else:
				fr_counter=0
				Alarm_on=False  

		if process_this_frame:
		        #Returns an array of bounding boxes of human faces in a image.
			face_locations=face_recognition.face_locations(rgb,model="hog")
			#Given an image, return the 128-dimension face encoding for the face
			face_encodings=face_recognition.face_encodings(rgb,face_locations)
			face_names=[]

			for face_encoding in face_encodings:
				matches=face_recognition.compare_faces(known_face_encoding,face_encoding,tolerance=0.5)
				#lower the tolerance more strict the comparisons i.e. error reduces
		
				name="Stranger"
				if True in matches:
					first_match_index=matches.index(True)
					name=known_face_names[first_match_index]

				face_names.append(name)

		process_this_frame=not process_this_frame

		for (top, right, bottom, left), name in zip(face_locations, face_names):
		        # Scale back up face locations since the frame we detected was scaled to 1/4 size
		        top *= 4
		        right *= 4
		        bottom *= 4
		        left *= 4

		        # Drawing rectangular box aroung the face
		        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

		        # Draw a label with a name below the face
		        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (255, 0, 0), cv2.FILLED)
		        font = cv2.FONT_HERSHEY_DUPLEX
		        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	                cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
			(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		    # Display the resulting image
		cv2.imshow('Video', frame)

		    # Hit 'q' on the keyboard to quit!
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break
	# Release webcam
	cap.release()
	cv2.destroyAllWindows()
