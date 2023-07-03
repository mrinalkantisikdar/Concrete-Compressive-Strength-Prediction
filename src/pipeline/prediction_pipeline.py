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
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl') # we have to write like this to run in linux 
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)        # only transform for test data

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData: # give all features except the target feature
    def __init__(self,
                 cement:float,
                 blast_furnace_slag:float,
                 water:float,
                 superplasticizer:float,
                 fine_aggregate:float,
                 age:float,
                 coarse_aggregate:float,
                 fly_ash:float):
        
        self.cement=cement
        self.blast_furnace_slag=blast_furnace_slag
        self.water=water
        self.superplasticizer=superplasticizer
        self.coarse_aggregate=coarse_aggregate
        self.fine_aggregate=fine_aggregate
        self.age=age
        self.fly_ash=fly_ash

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'cement':[self.cement],
                'blast_furnace_slag':[self.blast_furnace_slag],
                'fly_ash':[self.fly_ash],
                'water':[self.water],
                'superplasticizer':[self.superplasticizer],
                'coarse_aggregate':[self.coarse_aggregate],
                'fine_aggregate':[self.fine_aggregate],
                'age':[self.age]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)


