import cv2
import numpy as np 
import streamlit as st 
from PIL import Image,ImageEnhance

st.markdown(f'<p style="font-family:Georgia; text-align:center; color:#FF0000; font-size:40px; border-radius:2%;"> <b>Multipurpose Face App</b> </p>' , unsafe_allow_html=True)

@st.cache
def load_image(img):
	im = Image.open(img)
	return im

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
				 cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img, faces 


def detect_eyes(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
	for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def detect_smiles(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect Smiles
	smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img

def cartonize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Edges
	gray = cv2.medianBlur(gray, 5)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	#Color
	color = cv2.bilateralFilter(img, 9, 300, 300)
	#Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	return cartoon


def cannize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	canny = cv2.Canny(img, 100, 150)
	return canny

def main():
	activities = ["Detection","About"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Detection':
		st.subheader("Face Detection")

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file)
			# st.text("Original Image")
			# st.write(type(our_image))
			# st.image(our_image, width=300)

			enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring"])
			if enhance_type == 'Gray-Scale':
				st.text("Gray-Scaled Image")
				new_img = np.array(our_image.convert('RGB'))
				img = cv2.cvtColor(new_img,1)
				gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				# st.write(new_img)
				st.image(gray, width=300)

			elif enhance_type == 'Contrast':
				st.text("Contrast Image")
				c_rate = st.sidebar.slider("Contrast",0.0, 5.4, 0.3)
				enhancer = ImageEnhance.Contrast(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output, width=300)
				st.markdown(f'`Contrast strength is : {c_rate}`')
				

			elif enhance_type == 'Brightness':
				st.text("Bright Image")
				c_rate = st.sidebar.slider("Brightness",0.0,5.4, 0.3)
				enhancer = ImageEnhance.Brightness(our_image)
				img_output = enhancer.enhance(c_rate)
				st.image(img_output, width=300)
				st.markdown(f'`Brightness strength is : {c_rate}`')

			elif enhance_type == 'Blurring':
				st.text("Blur Image")
				new_img = np.array(our_image.convert('RGB'))
				blur_rate = st.sidebar.slider("Brightness",0.0,5.4, 0.3)
				img = cv2.cvtColor(new_img,1)
				blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
				st.image(blur_img, width=300)
				st.markdown(f'`Blur strength is : {blur_rate}`')

			elif enhance_type == 'Original':
				st.text("Original Image")
				st.image(our_image,width=300)
				
			else:
				# st.text("Original Image")
				st.image(our_image,width=300)


		# Face Detection and mage Effects
		task = ["Faces","Smiles","Eyes","Cannize","Cartonize"]
		feature_choice = st.sidebar.selectbox("Find Features",task)
		if st.button("Process"):

			if feature_choice == 'Faces':
				result_img,result_faces = detect_faces(our_image)
				st.image(result_img, width=300)

				st.success("Found {} faces".format(len(result_faces)))
			elif feature_choice == 'Smiles':
				result_img = detect_smiles(our_image)
				st.image(result_img, width=300)


			elif feature_choice == 'Eyes':
				result_img = detect_eyes(our_image)
				st.image(result_img, width=300)

			elif feature_choice == 'Cartonize':
				result_img = cartonize_image(our_image)
				st.image(result_img, width=300)

			elif feature_choice == 'Cannize':
				result_canny = cannize_image(our_image)
				st.image(result_canny,width=300)


	elif choice == 'About':
		# st.title("About App")
		st.markdown(f'<p style="font-family:Georgia; color:#FF0000; font-size:25px;"> <b>About App</b> </p>' , unsafe_allow_html=True)

		st.text("Build by using Streamlit and OpenCV")
		st.text("This helps to edit an image and detect different parts of a face.")
		st.text(" Built with 💖 by Akash Patel ")
		

if __name__ == '__main__':
		main()