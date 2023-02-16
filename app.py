import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app= Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

# prediction function 
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 12) 
    loaded_model = pickle.load(open("flight_risk_flask.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
    
def listToString(s):
   
    # initialize an empty string
    str1 = " "
   
    # return string 
    return (str1.join(s))
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Income more than 50K'
        else: 
            prediction ='Income less that 50K'            
        return render_template("result.html", prediction = prediction) 
        
@app.route('/predict',methods=['POST'])
def predict_fun():
	
    clf = pickle.load(open('flight_risk_flask.pkl','rb'))    
    cv = pickle.load(open('vector.pkl','rb'))

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        print(my_prediction)
	
    return render_template('result.html',input = request.form['message'], prediction = listToString(my_prediction))
        
if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=False)

