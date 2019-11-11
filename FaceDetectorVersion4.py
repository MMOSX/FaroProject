import cv2
import numpy as np
import imutils
from random import randint
#%%
tests_path = 'TestImages/rear/'
#%%
test_image = cv2.imread(tests_path + 't-pantaloni-rearCIE.jpg', cv2.IMREAD_GRAYSCALE)
#%% Cascade initializer
CASCADE="Face_cascade.xml"
FACE_CASCADE=cv2.CascadeClassifier(CASCADE)
#%% Define function
def detect_faces(image):

	#image=cv2.imread(image_path)
	image_grey=image.copy()#cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

	for x,y,w,h in faces:
	    sub_img=image[y-10:y+h+10,x-10:x+w+10]
	    #os.chdir("Extracted")
	    cv2.imwrite(str(randint(0,10000))+".jpg",sub_img)
	    #os.chdir("../")
	    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	cv2.imshow("Faces Found",image)
	if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
		cv2.destroyAllWindows()
#%%
# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate_bound(test_image, angle)
	detect_faces(rotated)
	#cv2.imshow("Rotated (Correct)", rotated)
	#cv2.waitKey(0)