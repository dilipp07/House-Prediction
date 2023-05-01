from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
from src.logger import logging
import os
import sys
import numpy as np


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    try:
        return render_template('index.html')
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    try:
        if request.method=='GET':
            return render_template('form.html')
        
        else:
            logging.info("entered predict_datapoint")
            data=CustomData(
                CRIM=float(request.form.get('CRIM')),
                ZN = float(request.form.get('ZN')),
                INDUS = float(request.form.get('INDUS')),
                CHAS = float(request.form.get('CHAS')),
                NOX = float(request.form.get('NOX')),
                RM = float(request.form.get('RM')),
                AGE = float(request.form.get('AGE')),
                DIS= float(request.form.get('DIS')),
                RAD = float(request.form.get('RAD')),
                TAX= float(request.form.get('TAX')),
                PTRATIO= float(request.form.get('PTRATIO')),
                B = float(request.form.get('B')),
                LSTAT = float(request.form.get('LSTAT'))

            )
            final_new_data=data.get_data_as_dataframe()

            
            print(final_new_data)
            print(np.array(final_new_data).reshape((1,-1)))
            # print(type(final_new_data))
            logging.info("data frame created")
            
            predict_pipeline=PredictPipeline()
            pred=predict_pipeline.predict(final_new_data)

            results=round(pred[0],2)

            return render_template('result.html',final_result=results)
    
    except Exception as e:
        logging.info("error occured in application")
        raise CustomException(e,sys)






if __name__=="__main__":
    app.run(host='0.0.0.0',port=5001
    ,debug=True)