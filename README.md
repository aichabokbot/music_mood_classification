# Music Mood Classification 

## Project Description
The objective is to build a music mood classification system on Pytorch using Deep Learning. The model would classify music tracks into 4 different labels: Happy, Sad, Angry and Relaxed. The model would learn patterns for each type of music to be able to detect the mood of new music tracks based on Spectograms. 

## Usage
#### Virtual environment setup

Run the following commands in the terminal to create the necessary environment. 
```bash
# Create a virtual environment
python3 -m venv audio_env
# Activate the virtual environment
source audio_env/bin/activate
# Install dependencies
make
```
**NOTE**: to run the whole code, you need to get access to the Spotify's API. To do so you need to put a `client_id` and `client_secret` inside a `credentials.py` file at the root of this repository. If you do not do it, you won't be able to create the spectograms and run the WebApp. 

## Data Generation

To do this project we collected the songs tagged in the LastFM Dataset. To download this data you can run the following command. 
```bash
make download_data
```
Then we applied the following steps for each song: 
- Download the .mp3 file with the Spotify's Web API
- Convert that .mp3 to a .wav file
- Create the spectogram from the .wav file

The notebook `build_spectograms.ipynb` shows how we did it with Python.

## Model
We used a Resnet18 CNN model (not pretrained) on the generated spectograms.

## Web Application
Using the library `streamlit`, we created a web application to make our model easy to apply. To launch the application, run (at root):
```bash
streamlit run app/app.py
```
**NOTE**: you need to have your Spotify's API credentials inside a `credentials.py` file at root. 
#### Instructions
- Search a song thanks to the Spotify's API **OR** upload your own song 
- The spectogram gets computed
- 1 or 2 moods are returned, depending on the song. The main mood will appear bigger.

## Sources
- Last FM Dataset: http://millionsongdataset.com/lastfm/
- Spotify's Web API: https://spotipy.readthedocs.io/en/2.16.1/

## Authors
- Aicha BOKBOT
- Arthur KRIEFF
- Eva FRANCOIS
- Corentin SENE