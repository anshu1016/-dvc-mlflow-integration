import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
from dotenv import load_dotenv
import os

from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse


# Now use the environment variable
load_dotenv()
MLFLOW_TRACKING_URI =  os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1)
    grid_search.fit(X_train,y_train)
    return grid_search # It will retuen the best model by tuning the model with all the params provided.

# Load the Prametere from params.yaml

params = yaml.safe_load(open('params.yaml'))['train']

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data = pd.read_csv(data_path)
    X=data.drop(columns=['Outcome']) # train data
    y=data['Outcome'] # output data
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Start the MLFlow run

    with mlflow.start_run():
        # Split the dataset into training and test dataset
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=random_state)
        signature=infer_signature(X_train,y_train)
        param_grid={
            'n_estimators':[100,200],
            'max_depth':[5,10,None],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,2]
        }
        # Perform Hyperparameter Tuning
        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and Evaluate the model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        print(f'Accuracy Score is: {accuracy}')

        # Log the Metrics
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_param("best_n_estimators: ",grid_search.best_estimator_.n_estimators)
        mlflow.log_param("best_max_depth: ",grid_search.best_estimator_.max_depth)
        mlflow.log_param("best_min_samples_split: ",grid_search.best_estimator_.min_samples_split)
        mlflow.log_param("best_min_samples_leaf: ",grid_search.best_estimator_.min_samples_leaf)


        # Log the confusion Metrix and classification Report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # Log the Both matrix and report in the form of log_text parameter
        mlflow.log_text(str(cm),'confusion_matrix.txt')
        mlflow.log_text(cr,'classification_report.txt')

        # Tracking URI typeStore
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name='Best Model _v1',
                input_example=X_train.iloc[:5],  # Use a small sample
                signature=signature
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                input_example=X_train.iloc[:5],
                signature=signature
            )


        # Create a directory to save the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)

        filename = model_path

        pickle.dump(best_model,open(filename,'wb'))

        print(f'Model saved to {model_path}')
        

if __name__=="__main__":
    train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])
# We are not using this parameters but just for the sake of understanding we are demonstrating here.
#  We are already define the parameters in our code.
# we are using only data and model_path