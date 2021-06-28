from flask import Flask, jsonify
import librosa
from tensorflow import keras
import numpy as np
import os
import sounddevice as sd
from scipy.io.wavfile import write
model = keras.models.load_model(os.path.join("D:/mini project/flask/","Emotion_Voice_Detection_Model.h5"))


app = Flask(__name__)

@app.route('/', methods=['GET'])

def predict():
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('examples2.wav', fs, myrecording)  # Save as WAV file 
    #file = request.files['file']
    data, sampling_rate = librosa.load('examples2.wav', res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=-1)
    x = np.expand_dims(x, axis=0)
    result = model.predict_classes(x)
    emotion = {0: 'neutral',
               1: 'calm',
               2: 'happy',
               3: 'sad',
               4: 'angry',
               5: 'fearful',
               6: 'disgust',
               7: 'surprised'}
    #return jsonify(result=str(result[0]))
    return jsonify(result=str(emotion[result[0]]))
    #return str(result)
@app.route('/record', methods=['POST','GET'])

def record():
  
    predict()
    return jsonify(result='Success')

if __name__ == '__main__':
    app.run(debug=True)