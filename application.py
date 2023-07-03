from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():        # this funciton will also be present in form.html
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            cement = float(request.form.get('cement')), # convert to float; not needed for catagorical data since catagorical data form drop down menu
            blast_furnace_slag= float(request.form.get('blast_furnace_slag')),
            water = float(request.form.get('water')),
            superplasticizer = float(request.form.get('superplasticizer')),
            coarse_aggregate = float(request.form.get('coarse_aggregate')),
            fine_aggregate = float(request.form.get('fine_aggregate')),
            age = float(request.form.get('age'))
            )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred)

        return render_template('results.html',final_result=results)     # return the results.html to form






if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000, debug=True)

