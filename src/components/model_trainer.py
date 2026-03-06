import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )


            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }


            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },

                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Linear Regression": {},

                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },

                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [.1, .01, .05],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, .05],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }


            model_report = {}


            for model_name, model in models.items():

                logging.info(f"Training model: {model_name}")

                param = params[model_name]

                if len(param) != 0:

                    grid = GridSearchCV(
                        model,
                        param,
                        cv=3,
                        scoring='r2',
                        n_jobs=-1
                    )

                    grid.fit(X_train, y_train)

                    model = grid.best_estimator_

                else:

                    model.fit(X_train, y_train)


                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)


                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)


                logging.info(f"{model_name} train score: {train_model_score}")
                logging.info(f"{model_name} test score: {test_model_score}")


                model_report[model_name] = test_model_score

                models[model_name] = model


            best_model_score = max(model_report.values())


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = models[best_model_name]


            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)


            logging.info(f"Best model found: {best_model_name}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)


            return r2_square


        except Exception as e:
            raise CustomException(e, sys)