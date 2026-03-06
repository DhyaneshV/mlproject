from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Data ingestion
obj = DataIngestion()
train_data, test_data = obj.initiate_data_ingestion()

# Data transformation
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

# Model training
modeltrainer = ModelTrainer()
modeltrainer.initiate_model_trainer(train_arr, test_arr)