import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            logging.info("done prediction")
            return pred
            logging.info("returned pred value")

            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
    CRIM:float,
    ZN:float,
    INDUS:float,
    CHAS:float,
    NOX:float,
    RM:float,
    AGE:float,
    DIS:float,
    RAD:float,
    TAX:float,
    PTRATIO:float,
    B:float,
    LSTAT:float):
        
        self.CRIM=CRIM,
        self.ZN=ZN,
        self.INDUS=INDUS,
        self.CHAS=CHAS,
        self.NOX=NOX,
        self.RM=RM,
        self.AGE=AGE,
        self.DIS=DIS,
        self.RAD=RAD,
        self.TAX=TAX,
        self.PTRATIO=PTRATIO,
        self.B=B,
        self.LSTAT=LSTAT
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'CRIM':[self.CRIM][0],
                'ZN':[self.ZN][0],
                'INDUS':[self.INDUS][0],
                'CHAS':[self.CHAS][0],
                'NOX':[self.NOX][0],
                'RM':[self.RM][0],
                'AGE':[self.AGE][0],
                'DIS':[self.DIS][0],
                'RAD':[self.RAD][0],
                'TAX':[self.TAX][0],
                'PTRATIO':[self.PTRATIO][0],
                'B':[self.B][0],
                'LSTAT':[self.LSTAT][0]}

            df = pd.DataFrame(custom_data_input_dict)
           
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
