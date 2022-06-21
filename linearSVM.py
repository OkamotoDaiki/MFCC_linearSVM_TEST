import glob
import soundfile
from subscript import FFTTool, MFCC
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

if __name__=="__main__":
    fpath = "recordings/*.wav"
    files = glob.glob(fpath)
    training_list = []
    label_list = []

    for fname in files:
        data, fs = soundfile.read(fname)
        data = FFTTool.ZeroPadding(data).process()
        window_data = np.hamming(len(data)) * data
        numChannels = 20
        cutpoint = 12
        mfcc = MFCC.MFCCclass(window_data , fs, cutpoint=12, numChannels=20)
        mfcc_array = mfcc.MFCC()
        mfcc_array = preprocessing.scale(mfcc_array) #normalization
        training_list.append(mfcc_array)
        label = fname.split('/')[1].split('_')[0]
        label_list.append(label)

    """linear SVM"""
    linear_svm = LinearSVC()
    scores = cross_val_score(linear_svm, training_list, label_list, cv=5)
    print("Cross-validation scores: {:.3f}".format(scores.mean()))
