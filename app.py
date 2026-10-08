from flask import Flask, render_template, request
import joblib, pandas as pd
app=Flask(__name__)
bundle=joblib.load("house_price_model.pkl")
model,columns=bundle["model"],bundle["columns"]
LOCATIONS=["Delhi","Mumbai","Bangalore","Pune","Hyderabad"]
@app.route("/",methods=["GET","POST"])
def home():
    prediction=None; error=None
    if request.method=="POST":
        try:
            row=pd.DataFrame([{"area_sqft":float(request.form["area"]),"bedrooms":int(request.form["bedrooms"]),"bathrooms":int(request.form["bathrooms"]),"age_years":int(request.form["age"]),"parking":int(request.form["parking"]),"location":request.form["location"]}])
            row=pd.get_dummies(row,columns=["location"]).reindex(columns=columns,fill_value=0)
            prediction=f"₹{model.predict(row)[0]:,.0f}"
        except: error="Please enter valid house details."
    return render_template("index.html",prediction=prediction,error=error,locations=LOCATIONS)
if __name__=="__main__": app.run(debug=True)

