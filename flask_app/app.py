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

# Behavior for root URL pattern
@app.route('/', methods=['GET', 'POST'])
def index():
    # Handles GET requests
    if request.method == 'GET':
        return render_template("index.html")

    # Handles POST requests
    else:
        # Extract input from request
        correct_label = request.form.get("correct_label")
        home_state = request.form.get("home_state")
        other_states = request.form.get("other_states")
        second_language = request.form.get("second_language")

        # Connect to database
        mysql.init_app(app)
        conn = mysql.connect()
        cursor = conn.cursor()

        # Insert values into database
        sql = "INSERT INTO responses VALUES(NULL, %s, %s, %s, %s)"
        cursor.execute(sql, (correct_label, home_state, other_states, second_language))
        conn.commit()

        # Display confirmation page
        return("<h1>Thank you for submitting!</h1>")

@app.route('/upload',methods=['POST'])
def upload(): 
    file=request.files['audio']
       
    print('the file is', file)
    filename=file.filename.split('.')[0]+'_new.'+file.filename.split('.')[-1]
    path = os.path.join(os.getcwd(), filename)
    
    # Save file to the path
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
