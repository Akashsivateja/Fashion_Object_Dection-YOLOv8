import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO

def app():
    st.set_page_config(page_title="Fashion Objects Detection", page_icon=":camera:", layout="wide")

    st.markdown("""
    <style>
    body {
        background-color: #33FFD1;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: #00698f;'>Fashion Objects Detection-Akash Jayampu</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #00698f;'>Powered by YOLOv8</h2>", unsafe_allow_html=True)

    model = YOLO('best.pt')
    object_names = list(model.names.values())

    col1, col2 = st.columns(2)

    with col1:
        st.write("Upload Image")
        uploaded_file = st.file_uploader("Select an image", type=['jpg', 'jpeg', 'png'])

    with col2:
        st.write("Detection Settings")
        select_all = st.checkbox("Select All")
        if select_all:
            selected_objects = object_names
        else:
            selected_objects = st.multiselect('Choose objects to detect', object_names)
        min_confidence = st.slider('Confidence score', 0.0, 1.0)

    submitted = st.button("Submit")

    if submitted:
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

            with st.spinner('Processing image...'):
                result = model(image)
                for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name = model.names[cls]
                    label = f'{object_name} {score}'

                    if model.names[cls] in selected_objects and score > min_confidence:
                        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(image, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_DUPLEX,0.5, (255, 0, 0), 2)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, caption='Detected objects', width=400)  

if __name__ == "__main__":
    app()
