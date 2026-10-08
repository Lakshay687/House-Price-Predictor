import pandas as pd, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
df=pd.read_csv("house_data.csv")
X=pd.get_dummies(df.drop(columns="price"),columns=["location"]); y=df.price
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42)
m=RandomForestRegressor(n_estimators=250,random_state=42).fit(a,c)
print("MAE:",mean_absolute_error(d,m.predict(b)))
print("R2:",r2_score(d,m.predict(b)))
joblib.dump({"model":m,"columns":list(X.columns)},"house_price_model.pkl")
