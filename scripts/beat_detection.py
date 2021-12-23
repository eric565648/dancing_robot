# import modules
import librosa 
import IPython.display as ipd 
import soundfile as sf

# read audio file 
x, sr = librosa.load('songs/Jingle_Bells_full.wav') 
# ipd.Audio(x, rate=sr)

# import modules
import madmom 

# approach 2 - dbn tracker
proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
act = madmom.features.beats.RNNBeatProcessor()('songs/Jingle_Bells_full.wav')
# act = madmom.features.beats.RNNBeatProcessor()('train/train1.wav')

beat_times = proc(act)
# print(beat_times)

clicks = librosa.clicks(beat_times, sr=sr, length=len(x))

sf.write('jingle_bell_ticks.wav', x+clicks, sr, 'PCM_24')