import syft as sy
import numpy as np
import numpy.typing as npt
from typing import Union, TypeVar, Any, TypedDict, TypeVar
import pandas as pd
from syft.service.policy.policy import MixedInputPolicy
from utils import check_status_last_code_requests

DataFrame = TypeVar("pandas.DataFrame")
NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
Dataset = TypeVar("Dataset", bound=tuple[NDArrayFloat, NDArrayInt])

class DataParamsDict(TypedDict):
    target: str
    ignored_columns: list[Any]

class ModelParamsDict(TypedDict):
    model: bytes
    n_base_estimators: int
    n_incremental_estimators: int
    train_size: float
    sample_size: int

DataParams = TypeVar("DataParams", bound=DataParamsDict)
ModelParams = TypeVar("ModelParams", bound=ModelParamsDict)


def ml_experiment(self, data: DataFrame, dataParams: DataParams, modelParams: ModelParams) -> dict:
    # preprocessing
    print("Inside ml_experiment")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import cloudpickle
    import pickle

    def preprocess(data: DataFrame) -> tuple[Dataset, Dataset]:

        # Step 1: Prepare the data for training
        # Drop rows with missing values in Q1
        data = data.dropna(subset=[dataParams["target"]])

        # Separate features and target variable (Q1)
        # TODO: ignorar também patient_id?
        y = data[dataParams["target"]]
        X = data.drop(dataParams["ignored_columns"], axis=1)

        # Step 2: Split the data into training and testing sets
        X_train, _, y_train, _ = train_test_split(X, y, train_size=modelParams["train_size"], stratify=y, random_state=42)

        return X_train, y_train

    def train(model, training_data: tuple[pd.DataFrame, pd.Series]) -> RandomForestClassifier:
        X_train, y_train = training_data
        model.fit(X_train, y_train)
        return model
    
    # Preprocess data
    training_data = preprocess(data)
    print(f"Training data shape: {training_data[0].shape}")
    if modelParams["model"]:
        model = modelParams["model"]
        clf = pickle.loads(model)
        clf.n_estimators += modelParams["n_incremental_estimators"]
    else:
        clf = RandomForestClassifier(random_state=42, n_estimators=modelParams["n_base_estimators"], warm_start=True)
    
    clf = train(clf, training_data)

    return {"model": cloudpickle.dumps(clf), "n_base_estimators": modelParams["n_base_estimators"], "n_incremental_estimators": modelParams["n_incremental_estimators"], "train_size": modelParams["train_size"], "sample_size": len(training_data[0])}


class FLClient:
    def __init__(self):
        self.datasites = {}
        self.dataParams = {}
        self.modelParams = {}
        self.fl_epochs = 10
    
    def add_client(self, name, url, email, password):
        try:
            client = sy.login(email=email, password=password, url=url)
            self.datasites[name] = client
            print(f"Successfully connected to {name} at {url}")
        except Exception as e:
            print(f"Failed to connect to {name} at {url}: {e}")

    def check_status(self):
        """
        Checks and prints the status of all connected silos.
        """
        for name, client in self.datasites.items():
            try:
                datasets = client.datasets
                print(f"{name}:  Connected ({len(datasets)} datasets available)")
            except Exception as e:
                print(f"{name}: Connection failed ({e})")

    def set_data_params(self, data_params):
        self.dataParams = data_params
        print(f"Data parameters set: {data_params}")

    def set_model_params(self, model_params):
        self.modelParams = model_params
        print(f"Model parameters set: {model_params}")

    def get_data_params(self):
        return self.dataParams

    def get_model_params(self):
        return self.modelParams


    def send_request(self):

        if not self.datasites:
            print("No clients connected. Please add clients first.")
            return
        
        if self.dataParams is None or self.modelParams is None:
            print("DataParams and ModelParams must be set before sending the request.")
            return
        
        for site in self.datasites:
            data_asset = self.datasites[site].datasets[0].assets[0]
            client = self.datasites[site]
            syft_fl_experiment = sy.syft_function(
                input_policy=MixedInputPolicy(
                    client=client,
                    data=data_asset,
                    dataParams=dict,
                    modelParams=dict
                )
            )(ml_experiment)
            ml_training_project = sy.Project(
                name="ML Experiment for FL",
                description="""Test project to run a ML experiment""",
                members=[client],
            )
            ml_training_project.create_code_request(syft_fl_experiment, client)
            project = ml_training_project.send()


    def check_status_last_code_requests(self):
        """
        Display status message of last code request sent to each datasite.
        """
        check_status_last_code_requests(self.datasites)


    def run_model(self):
        modelParams = self.get_model_params()
        dataParams = self.get_data_params()
        for epoch in range(self.fl_epochs):
            print(f"\nEpoch {epoch + 1}/{self.fl_epochs}")

            for name, datasite in self.datasites.items():
                print(f"Training on {name}...")

                data_asset = datasite.datasets[0].assets[0]
                print(f"Data asset: {data_asset}")
                print(f"Data params: {dataParams}")
                print(f"Model params: {modelParams}")
                modelParams = datasite.code.ml_experiment(
                    data=data_asset, dataParams=dataParams, modelParams=modelParams
                ).get_from(datasite)

                print(f"Model trained on {name} with {modelParams['n_base_estimators']} base estimators and {modelParams['n_incremental_estimators']} incremental estimators")
        self.set_model_params(modelParams)
