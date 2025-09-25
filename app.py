from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle

app = Flask(__name__)

def get_clean_data(form_data):
    gestation = float(form_data["gestation"])
    parity = int(form_data["parity"])
    age = float(form_data["age"])
    height = float(form_data["height"])
    weight = float(form_data["weight"])
    smoke = float(form_data["smoke"])

    cleaned_data = {"gestation" : [gestation],
                    "parity": [parity],
                    "age": [age],
                    "height": [height],
                    "weight": [weight],
                    "smoke": [smoke]}


    return cleaned_data

@app.route("/", methods = ["GET"])
def home():
    return render_template("index.html")


#deffine endpoint
@app.route('/predict',methods = ['POST'])
def get_prediction():
    #baby_data = request.get_json() # this will extract the json data provided in the frontend.
    form_data = request.form # this will get data from a form fromt end!
    baby_data_cleaned = get_clean_data(form_data)
    
    #convert to dataframe
    baby_df = pd.DataFrame(baby_data_cleaned)
    #When getting data from a form.
    

    #load trained ML model
    with open('model/model.pkl','rb') as obj:
        model = pickle.load(obj)

    #make predictions on user data.
    prediction = model.predict(baby_df)

    prediction =  round(float(prediction),2)
    # return response in json format.
    # responce =  {'Prediction':prediction}
    # return jsonify(responce, "Ounce")
    return render_template("index.html",prediction = prediction) # this will return responce to the front end placeholder.





if __name__ =='__main__':
    app.run(debug =True)