from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import send_from_directory
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload',methods=['POST'])
def upload(): 
    file=request.files['audio']
       
    print('the file is', file)
    filename=file.filename.split('.')[0]+'_new.'+file.filename.split('.')[-1]
    path = os.path.join(os.getcwd(), filename)
    
    #Save file to the path
    file.save(path)
    print('GET=',file.filename)
    print('UPLOAD=',filename,'#'*50)
    return jsonify({"path":path})


@app.route('/download/<filename>')
def download(filename):
    print('DOWNLOAD=',filename,'*'*50)
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


if __name__ == '__main__':
    app.run()