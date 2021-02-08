from credentials import client_id, client_secret
import pandas as pd
import requests
import time
import os
import spotipy
from tqdm import tqdm
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def create_client_spotify(client_id:str, client_secret:str):
    """
    Function that creates a spotify client
    Input:
        - client_id: crediential
        - client_secret: credential
    Output:
        - sp: spotify client object
    """
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, 
                                                      client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp


def search_track_spotify(query:str):
    """
    Function that searches a song from Spotify API
    Input:
        - query: song and artist
    Output:
        - nested dictionary of the results of the query
    """
    sp = create_client_spotify(client_id, client_secret)
    try:
        results = sp.search(q=query, type="track", limit=1, offset=0)
    except:
        results = {}
    return results


def get_preview_url(results:dict):
    """
    Function that fetches the preview_url from the results of the query passed on spotify
    Input:
        - results: results of the query from Spotify API
    Output:
        - preview_url (str)
    """
    if results == {}:
        return None
    
    for item in results["tracks"]["items"]:
        if item["preview_url"] != None:
            return item["preview_url"]
        else:
            continue

            
def download_song(url:str, song_id:str, dest_dir:str):
    """
    Function that download a song based on its url
    Input:
        - url: url of the song
        - song_id: id of the song based on the last fm dataset
        - dest_dir: directory destination (ends with "/")
    """
    if url == None:
        return
    
    if os.path.exists(dest_dir + song_id + ".mp3"):
        return
    
    response = requests.get(url, stream=True)
    if response.status_code == 200: 
        open(dest_dir + song_id + '.mp3', 'wb').write(response.content)
        time.sleep(0.01)
    elif response.status_code == 429:
        print('Spotify API: 429 - Too many requests. Waiting 60 seconds.')
        time.sleep(60)
        state = False
    else:
        print('Spotify API: Unspecified error. No response was received. Trying again after 60 seconds...')
        time.sleep(60)
        

def get_all_songs_from_df(df:pd.DataFrame, dest_dir:str):
    """
    Function that downloads all the songs mentionned in df in the directory mp3_preview 
    Input:
        - df: assume schema is the following {track_id:str, artist:str, title:str}
        - dest_dir: directory destionation (ends with "/")
    """
    sp = create_client_spotify(client_id, client_secret)
    
    # Check if directory is already created
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    
    # Download all songs in dataset
    for row in tqdm(range(df.shape[0])):
        query = df.loc[row, "artist"] + " " + df.loc[row, "title"]
        song_id = df.loc[row, "track_id"]
        results = search_track_spotify(query)
        url = get_preview_url(results)
        download_song(url, song_id, dest_dir)