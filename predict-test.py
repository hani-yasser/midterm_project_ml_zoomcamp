
#!/usr/bin/env python
#coding: utf-8

import requests


url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
customer = {
     "age": 71, 
     "sex": 0, 
     "cp": 0, 
     "trestbps": 112, 
     "chol": 149, 
     "fbs": 0, 
     "restecg": 1, 
     "thalach": 125, 
     "exang": 0, 
     "oldpeak": 1.6, 
     "slope": 1, 
     "ca": 0, 
     "thal": 2, 
     
}

response = requests.post(url, json=customer).json()
print(response)

if response['target'] == True:
    print('The patient has heart Disease %s' % customer_id)
else:
    print('The patient has no heart Disease %s' % customer_id)