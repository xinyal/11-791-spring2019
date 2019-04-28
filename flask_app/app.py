from flask import Flask
from flask import render_template

from flaskext.mysql import MySQL
from flask import request
from flask import jsonify
from flask import send_from_directory
import os


app = Flask(__name__)

# Configure database connection
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = '11-791project'
app.config['MYSQL_DATABASE_DB'] = 'dialect_info'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
conn = mysql.connect()
cursor = conn.cursor()

# Behavior for root URL pattern
@app.route('/')
def index():
    cursor.execute("INSERT INTO responses VALUES(0, \"Yes\", \"California\", \"Texas\", \"No\")")
    conn.commit()
    print("hello")
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
