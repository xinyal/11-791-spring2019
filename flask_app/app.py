from flask import Flask
from flask import render_template
from flaskext.mysql import MySQL

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


if __name__ == '__main__':

    app.run()
