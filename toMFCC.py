import sys
import librosa
import numpy as np
import os

USAGE = """
    USAGE: python feature.py <filename.wav>  
    e.g. "python feature.py "e0.wav" outputs its MFCC features into the /MFCC folder, with the same filename and a .npy extension
"""

print (sys.argv)
try:
    FILENAME = sys.argv[1]
    DESTPATH = sys.argv[2] # e.g. "speech"
except Exception:
    print (USAGE)
    sys.exit(0)

x, fs = librosa.load(FILENAME)
mfccs = librosa.feature.mfcc(x, sr = fs, hop_length = int(fs/100), n_fft = int(fs/40), n_mfcc = 40) # https://github.com/librosa/librosa/issues/584
#np.save(DESTPATH, np.array(mfccs))
np.save(DESTPATH + '/' + os.path.splitext(os.path.basename(FILENAME))[0] + '.npy', np.array(mfccs))
print (np.array(mfccs).shape)
#np.dstack((mfcc_left, mfcc_right, mfcc_mono))) #https://github.com/keunwoochoi/embedding/blob/master/main_prepare.py
print("Done: %s, mfcc" % str(FILENAME))
