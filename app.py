#To make the user interface with streamlit
import streamlit as st 
from PIL import Image 
import numpy as np 
import main

st.title("Object Detection with Ultralitic's YoloV8")

st.divider()

st.header("Change utility by switching tabs")

live_vid, snapshot, pic,video = st.tabs(["Live Video","Take a Photo","Upload a pic","Upload a video"])

#Taking pictures
with snapshot:
    st.header("Click to snap.")
    img = st.camera_input("Take a Picture")
    #converting to PIL then to numpy array
    img = Image.open(img)
    img  = np.array(img)
    with st.spinner("Loading..."):
        main.image_processor(img)
    st.success("Your Detection")
    
#Uploaded pictures
with video:
    st.header("Upload a video")
    vid = st.file_uploader(" ")
    with st.spinner("Loading..."):
        main.video_processor(vid)
    st.success("Your Detection")
#
with pic:
    st.header("Please upload a picture")
    img = st.file_uploader(" ")
    #converting to PIL then to numpy array
    img = Image.open(img)
    img  = np.array(img)
    with st.spinner("Loading..."):
        main.image_processor(img)
    st.success("Your Detection")

with live_vid:
    st.header("Working on it")
    
#TODO: Figure out how to make the programs communicate information. 