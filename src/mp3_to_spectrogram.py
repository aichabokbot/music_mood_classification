from os import path
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import pylab
import numpy as np


class Spectrogram:
    def __init__(self):
        pass

    def mp3_to_wav(self, src, dest):
        sound = AudioSegment.from_mp3(src)
        sound.export(dest, format="wav")

    def wav_to_spectogram(self, src, dest):
        y, sr = librosa.load(src)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        
        x_pixels = 384
        y_pixels = 128
        plt.figure(figsize=(x_pixels/100, y_pixels/100))
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')

        pylab.savefig(dest, bbox_inches=None, pad_inches=0, dpi=100)
        pylab.close()
        
if __name__ == '__main__':

    # convert to jpg
    Spectrogram().mp3_to_wav('../data/mp3/test.mp3', '../data/wav/test.wav')
    Spectrogram().wav_to_spectogram('../data/wav/test.wav','../data/spectogram/test.jpg')
   


