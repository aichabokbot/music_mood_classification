import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import cv2 
import glob
import os
import time
from src.mp3_to_spectrogram import Spectrogram
from src.spotify_utils import *
from src.pytorch_utils import predict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

def clean_init():
    """
    Function to clean the previously generated files (if any)
    """
    files = ["myfile.wav", "myspec.jpg"]
    for file in files:
        try:
            os.remove(file)
        except FileNotFoundError:
            continue
            
def compute_spectrogram():
    with st.spinner(text='Spectogram is being computed...'):
        spec = Spectrogram()
        spec.wav_to_spectogram("myfile.wav", "myspec.jpg")
        spectrogram = cv2.imread("myspec.jpg")
        st.image(spectrogram, caption="spectrogram", width=WIDTH, channels='BGR')
        st.success('Spectogram Computed')

os.chdir(os.path.dirname(__file__))
### HEADER ####
st.write("""
# Mood Detection Algorithm
Goal is to detect the mood of a given song
""")

image = cv2.imread("first_image.png")
WIDTH = 700
st.image(image, caption="Music Detection", width=WIDTH, channels='BGR')
### END HEADER ###

clean_init()
option = st.radio('Select', ["Spotify Search", "Upload Song"])
state = False

### SPECTOGRAMS ###

if option == "Upload Song":
    file = st.file_uploader("Upload Song")
    if file != None:
        st.audio(file)
        data = file.read()
        with open('myfile.wav', mode='wb') as f:
            f.write(data)
        compute_spectrogram()
        state = True
        time.sleep(2)
        
elif option == "Spotify Search":
    artist = st.text_input('Enter Artist')
    song = st.text_input("Enter Song")
    if artist!="" or song!="":
        query = artist + " " + song
        time.sleep(1)
        with st.spinner(text='Searching...'):
            result = search_track_spotify(query)
            url = get_preview_url(result)
            if url != None:
                name = result["tracks"]["items"][0]["name"]
                st.success("Song found, playing "+name)
                st.audio(url)
                response = requests.get(url, stream=True)
                open('myfile.wav', 'wb').write(response.content)
                compute_spectrogram()
                state = True
            else:
                st.text("No Preview Available")
                state = False
        time.sleep(2)

### MOOD DETECTION ###
if state == True:
    st.write("""
    ## Music Mood(s)
    """)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    # Change the last layer shape
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    # Put trained weights
    model.load_state_dict(torch.load("Resnet_SGD_valscore_60.pt", map_location=torch.device(device) ))
    model.eval()
    
    ## Predict
    class_names = ["angry", "happy", "relaxed", "sad"]
    loader = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # Display Mood
    with st.spinner(text='Detecting Mood(s)...'):
        proba, mood = predict(model, "myspec.jpg", loader, class_names)
        image0 = cv2.imread(mood[0]+".png")
        image1 = cv2.imread(mood[1]+".png")
        p0 = proba[0]/proba.sum()
        p1 = proba[1]/proba.sum()
        
        if p1 > 0:
            col1, col2 = st.beta_columns((p0, p1))
            col1.image(image0, caption=mood[0], width=int(p0*WIDTH), channels='BGR')
            col2.image(image1, caption=mood[1], width=int(p1*WIDTH), channels='BGR')
            st.success("Done")
        else:
            st.image(image0, caption=mood[0], width=WIDTH, channels='BGR')
        state = 'END'
    
if state == 'END':
    st.header("Thank you for using our WebApp!")
    