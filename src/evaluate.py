import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
from dotenv import load_dotenv
import os


# Now use the environment variable
load_dotenv()
MLFLOW_TRACKING_URI =  os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')


# Load the Prametere from params.yaml
params = yaml.safe_load(open('params.yaml'))['evaluate']
print('checking Params: ',params)

def evaluate(data_path,model_path):
    print('Checking Arguments: ',data_path, model_path)
    data = pd.read_csv(data_path)
    X=data.drop(columns=['Outcome'])
    y=data['Outcome']

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load the model from the disk
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    # Log metrics to MLFlow
    mlflow.log_metric('accuracy', accuracy)
    print('Model Accuracy: ' ,accuracy)


if __name__=="__main__":
    evaluate(params['data'],params['model'])