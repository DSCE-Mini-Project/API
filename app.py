from flask import Flask, jsonify
import librosa
from tensorflow import keras
import numpy as np
import os
model = keras.models.load_model(os.path.join("D:/mini project/flask/","Emotion_Voice_Detection_Model.h5"))


app = Flask(__name__)

@app.route('/<filename>', methods=['POST','GET'])

def predict(filename):
    
    #file = request.files['file']
    data, sampling_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=-1)
    x = np.expand_dims(x, axis=0)
    result = model.predict_classes(x)
    
    #return jsonify(result=str(result[0]))
    return jsonify(result=str(result[0]))
    #return str(result)

if __name__ == '__main__':
    app.run(debug=True)