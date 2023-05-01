import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

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
            return pred
            

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
        self.LSTAT=LSAT
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'CRIM':[self.CRIM],
                'ZN':[self.ZN],
                'INDUS':[self.INDUS],
                'CHAS':[self.CHAS],
                'NOX':[self.NOX],
                'RM':[self.RM],
                'AGE':[self.AGE],
                'DIS':[self.DIS],
                'TAX':[self.TAX],
                'PTRATIO':[self.PTRATIO],
                'B':[self.B],
                'LSAT':[self.LSAT]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
