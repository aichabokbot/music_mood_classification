import requests
import zipfile
import os
import glob
import pandas as pd
import json
import shutil
import sys

TRAIN = "http://millionsongdataset.com/sites/default/files/lastfm/lastfm_train.zip"
TEST = "http://millionsongdataset.com/sites/default/files/lastfm/lastfm_test.zip"

#Declare the lists 
list_happy = ['happy', 'happiness', 'joyous', 'bright',
             'cheerful', 'humorous', 'fun', 'merry', 
              'exciting', 'silly']

list_angry = ['angry', 'aggressive', 'outrageous', 'fierce',
             'anxious', 'rebellious', 'tense', 'fiery',
             'hostile', 'anger']

list_sad = ['sad', 'bittersweet', 'bitter', 'tragic',
            'depressing', 'sadness', 'gloomy', 'miserable',
            'funeral', 'sorrow']

list_relaxed = ['relaxed', 'tender', 'soothing', 'peaceful',
               'gentle', 'soft', 'quiet', 'calm',
               'mellow', 'delicate']

list_moods = list_happy + list_angry + list_sad + list_relaxed
set_moods = set(list_moods)

def download_data(url:str, dest:str):
    response = requests.get(url, allow_redirects=True)
    dl = 0
    total_length = response.headers.get('content-length')
    total_length = int(total_length)
    print("Started downloading...")
    for data in response.iter_content(chunk_size=4096):
        dl += len(data)
        open(dest, 'wb').write(data)
        done = int(50 * dl / total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)))
        sys.stdout.flush()
    print("Content downloaded!")

def unzip(file_path:str, dest:str):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest)
    os.remove(file_path)

def mood_tag(my_list):
    """
    Define a function which return the corresponding mood tag for a given tags list
    Input : 
        - my_list: list of all tags related to a song (not necessarily mood tags - can also be type such as rock, pop etc)
    Output:
        - tag: mood tag corresponding to a song (which is including in the 20 tags of the list_moods)
    """
    set_list = set(my_list)
    tag_set = set_moods & set_list
    tag = list(tag_set)[0]
    return (tag)

def mood_cat(tag):
    """
    Define a function which return the corresponding mood category
    Input : 
        - tag: mood tag corresponding to a song (which is including in the 20 tags of the list_moods)
    Output:
        - tag_cat: category high level of the mood tag (among 'Happy', 'Sad', 'Angry' & 'Relaxed')
    """
    tag_cat = ''
    if tag in list_happy:
        tag_cat =  'Happy'
    elif tag in list_sad:
        tag_cat = 'Sad'
    elif tag in list_angry:
        tag_cat = 'Angry'
    elif tag in list_relaxed:
        tag_cat = 'Relaxed'
    return tag_cat


def json_to_dataframe(directory):
    """
    Function that creates a dataframe containing only the songs that have mood tags 
    Input:
        - directory: directory where all the json files are stored (with "/")
    Output:
        - df: dataframe containing only the songs that have mood tags
    """
    # Get all files
    path_json = directory + "**/*.json"
    files = glob.glob(path_json, recursive=True)
    
    # Find the index that contain mood tags
    index_list = []
    for i, file in enumerate(files):
        file_dict = json.load(open(files[i]))
        tags_list = [item[0] for item in file_dict['tags']]
        set_tags = set(tags_list)
        set_moods = set(list_moods)
    
        #This only keep the index where there is one tag in common (and no more)
        if (len(set_tags & set_moods) == 1) :
            index_list.append(i)  
    
    #Create DataFrame with the index chosen
    df = pd.DataFrame(columns=["artist", "timestamp", "tags", 
                               "track_id", "title", "tags_list"])
    for i in range(len(index_list)):
        idx = index_list[i]
        file_dict = json.load(open(files[idx]))
        tags_list = [item[0] for item in file_dict['tags']]
        file_dict.update({'tags_list': tags_list})
        file_dict.pop("similars", None)
        for key, val in file_dict.items():
            df.at[i, key] = val
    
    df['mood_tag'] = df['tags_list'].apply(lambda x: mood_tag(x))
    df['mood_cat'] = df['mood_tag'].apply(lambda x: mood_cat(x))
    
    return df

if __name__ == "__main__":
    download_data(TRAIN, dest="data/train.zip")
    os.mkdir("data/train")
    unzip("data/train.zip", dest="data/train")
    df_train = json_to_dataframe("data/train/")
    df_train.to_csv("train.csv")
    shutil.rmtree('data/train/')
    
    download_data(TEST, dest="data/test")
    os.mkdir("data/test")
    unzip("data/test.zip", dest="data/test")
    df_test = json_to_dataframe("data/test/")
    df_test.to_csv("test.csv")
    shutil.rmtree('data/test/')