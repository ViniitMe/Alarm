import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
import playsound
import time
import datetime

def main():
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

	cap=cv2.VideoCapture(0)

	while (1):
		ret,frame=cap.read()
		frame=imutils.resize(frame,width=500)
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
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
			(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		cv2.imshow("video",frame)

		if cv2.waitKey(1) & 0xFF==ord('q'):
			break

	cv2.destroyAllWindows()
	cap.release()


