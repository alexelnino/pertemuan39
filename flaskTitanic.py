from flask import Flask, render_template
from sklearn.externals import joblib

app = Flask(__name__)

# @app.route('/')
# def home():
#     return ('<h1>'
#     +str(model.predict([[0, 1, 0, 1, 0, 2, 23, 1, 2, 200, 1, 0]]))+
#     '</h1>')

@app.route('/')
def welcome():
    return render_template('home.html')

if __name__ == '__main__' :
    model= joblib.load('modelTitanic')
    app.run(debug=True)