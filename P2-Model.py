import pandas as pd
import pickle
import sys
import os

def predict_price(Year, Mileage, State, Make, Model):

    clf =  pickle.load(open(os.path.dirname(__file__) + '/CarPricing.pkl', 'rb'))
    url_ = pd.DataFrame([url], columns=['url'])
  
    # Create features

    # Make prediction
    p1 = clf.predict_price()

    return p1

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add parameters')
        
    else:

        url = sys.argv[1]

        p1 = predict_proba(Year, Mileage, State, Make, Model)
        
        print(Year)
        print(Mileage)
        print(State)
        print(Make)
        print(Model)
        print('Price of the car is: ', p1)
