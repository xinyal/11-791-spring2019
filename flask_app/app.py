from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import send_from_directory
import os

# our model
from src.toMFCC import get_mfcc
from src.pred_func import pred
from dialect import predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload',methods=['POST'])
def upload(): 
    #get the audio file
    file=request.files['audio']
       
    print('the file is', file)
    #rename the file
    filename=file.filename.split('.')[0]+'_new.'+file.filename.split('.')[-1]
    path = os.path.join(os.getcwd(), filename)
    
    #Save file to the path
    file.save(path)
    print('GET=',file.filename)
    print('UPLOAD=',filename,'#'*50)

    # extract features
    feat_filepath = get_mfcc(path, '.')
    print('FEATURES EXTRACTED:', feat_filepath)

    # send features as test to model
    # Xinya's model
    label1 = pred(feat_filepath)
    # Xianyang's model
    label2 = predict(feat_filepath)
    print(label1, label2)

    return label1


@app.route('/get',methods=['GET'])
def download(number):
    print('get the number', number)
    return send_from_directory(app.config['UPLOAD_FOLDER'],number)



if __name__ == '__main__':
    app.run()