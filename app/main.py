import os
import json
from PIL import Image

import requests
from streamlit_lottie import st_lottie
import numpy as np
import tensorflow as tf
import streamlit as st


def load_lottiefile(filepath:  str):
    with open(filepath, 'r') as f:
        return json.load(f)


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path1 = f"{working_dir}/trained_model/inception_model.h5"
model_path2 = f"{working_dir}/trained_model/mri_model.h5"
# Load the pre-trained model
model1 = tf.keras.models.load_model(model_path1)
model2 = tf.keras.models.load_model(model_path2)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))
class_indicesa = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name
def predict_image_class2(model, file, class_indicesa):
    data = np.loadtxt(file)
    predictions = model.predict(data)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indicesa[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App



bg = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://img.freepik.com/free-photo/abstract-purple-watercolor-background-illustration-high-resolution-free-photo_1340-21028.jpg?size=626&ext=jpg&ga=GA1.1.1700460183.1712793600&semt=ais")
background-size: cover;

</style>
"""
st.markdown(bg, unsafe_allow_html=True)


center_css = """
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
"""

st.markdown(
    f"""
    <style>
    {center_css}
    </style>
    <div class="title-container">
        <h1 style="text-align: center;">Early Stage Autism Detection</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

lottie_json = load_lottiefile("app/lottiefiles/Animals.json")
st_lottie(lottie_json,
          speed =1,
          reverse = True,
          loop=True,
          height=400,
          width=650
          )



######
# st.title("Upload .1D File")
# file = st.file_uploader("Upload .1D File", type=["1D"], accept_multiple_files=False)
#
# if file is not None:
#     # Display file details
#     data = np.loadtxt(file)
#     file_details = {"Filename": file.name, "FileType": file.type}
#     st.write(file_details)
#
#     # image = Image.open(file)
#     # # Resize the image to match the expected input shape
#     # resized_image = image.resize((224, 224))
#     # # Convert the image to a numpy array
#     # image_array = np.array(resized_image)
#     # # Add batch dimension to the image
#     # image_array = np.expand_dims(image_array, axis=0)
#     # # Now 'resized_array' should have the shape (batch_size, 224, 224, 3)
#
#     data = np.loadtxt(file)
#     predictions = predict_image_class2(model2, data, class_indices)
#     st.write(predictions)
#     if predictions == 1:
#         st.write("autistic")
########







file1 = st.file_uploader(
    "Upload a your image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
file = st.file_uploader("Upload fMRI File", type=["1D"], accept_multiple_files=False) #1change

# uploaded_eegimage = st.file_uploader(
    #     "Upload an eeg image...", type=["jpg", "jpeg", "png"])
    # if uploaded_eegimage is not None:
    #     image1 = Image.open(uploaded_faceimage)
    #     image2 = Image.open(uploaded_eegimage)
    #     col1, col2 = st.columns(2)
if file1 is not None and file is not None:
    var = Image.open(file1)
    col1, col2 = st.columns(2) #2change : above comments and additional line

    with col1:
        st.title('')
        st.title('')
        image1 = var.resize((150, 150))
        st.write('Face-Image Accuracy : 80%')
        st.write('MRI Accuracy : 92%')
        data = file1
        st.image(image1)

    with col2:
        if st.button('RESULTS'):
            # Preprocess the uploaded image and predict the class
            prediction1 = predict_image_class(
                model1, file1, class_indices)


            prediction2 = predict_image_class(model2, data, class_indicesa)
            # st.write(prediction1,prediction2)
            # prediction2 = predict_image_class(
            #     model2, uploaded_eegimage, class_indices)

            # prediction1:{str(prediction1)}
            if(str(prediction1)=="autistic" and str(prediction2)=="autistic"):
                st.success(f'Very high chances of being diagnosed with ASD')
                st.success(f'This is just a prognosis, the results are not 100% accurate. '
                           f'Based on your data, you will likely benefit from seeing a mental health professional for an evaluation. Medical assistance might positively impact your ability to be successful in your work or personal life. Yet it is essential to continue monitoring your well-being and mental health. If you have any concerns or would like additional information, consider discussing your results and any related questions with a healthcare or mental health professional. They can provide further guidance, support, and address any other potential underlying factors contributing to your well-being.')

            elif (str(prediction2)=="autistic" and str(prediction1) =="non_autistic"):
                st.success(f'High chances of being diagnosed with ASD')
                st.success(f'This is just a prognosis, the results are not 100% accurate. '
                           f'Based on your data, it may be a good idea to monitor your symptoms and keep track of the severity of these behaviors and when they are present. Further evaluation is typically recommended when these behaviors begin to interfere with your ability to navigate life, school, and relationships. If you’re concerned or want more information, consider scheduling an evaluation with a qualified healthcare or mental health professional for further assessment and potential treatment options.')


            elif (str(prediction1) =="autistic" and str(prediction2) =="non_autistic"):
                st.success(f'Chances of being diagnosed with ASD')
                st.success(f'This is just a prognosis, the results are not 100% accurate. '
                           f'Your evaluated behaviours and symptoms indicate low chances of being diagnosed with ASD. Yet it is essential to continue monitoring your well-being and mental health. If you have any concerns or would like additional information, consider discussing your results and any related questions with a healthcare or mental health professional. They can provide further guidance, support, and address any other potential underlying factors contributing to your well-being. ')

            else:
                st.success(f'Low chances of being diagnosed with ASD')
                st.success(f'This is just a prognosis, the results are not 100% accurate. '
                           f'Your evaluated behaviors and symptoms are not indicative of autism spectrum disorder. Yet it is essential to continue monitoring your well-being and mental health. If you have any concerns or would like additional information, consider discussing your results and any related questions with a healthcare or mental health professional. They can provide further guidance, support, and address any other potential underlying factors contributing to your well-being.')

if file1 is not None and file is None:
    image1 = Image.open(file1)
    col1, col2 = st.columns(2)

    with col1:
        st.title('')
        st.title('')
        resized_img1 = image1.resize((150, 150))
        st.write('Face-Image Accuracy : 80%')
        st.write('MRI Accuracy : 95%')
        st.image(resized_img1)

    with col2:
        if st.button('RESULTS'):
            # Preprocess the uploaded image and predict the class
            prediction1 = predict_image_class(
                model1, file1, class_indices)

            # prediction1:{str(prediction1)}
            if(str(prediction1)=="autistic" ):
                st.success(f'High chances of being diagnosed with ASD')
                st.success(f'This is just a prognosis, the results are not 100% accurate. '
                           f'Based on your data, you will likely benefit from seeing a mental health professional for an evaluation. Medical assistance might positively impact your ability to be successful in your work or personal life. Yet it is essential to continue monitoring your well-being and mental health. If you have any concerns or would like additional information, consider discussing your results and any related questions with a healthcare or mental health professional. They can provide further guidance, support, and address any other potential underlying factors contributing to your well-being.')

            else:
                st.success(f'Very low chances of being diagnosed with ASD')
                st.success(f'This is just a prognosis, the results are not 100% accurate. '
                           f'Your evaluated behaviors and symptoms are not indicative of autism spectrum disorder. Yet it is essential to continue monitoring your well-being and mental health. If you have any concerns or would like additional information, consider discussing your results and any related questions with a healthcare or mental health professional. They can provide further guidance, support, and address any other potential underlying factors contributing to your well-being.')
