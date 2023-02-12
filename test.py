import torch
import mlflow

model = mlflow.pytorch.load_model('runs:/021906e767b44b87b915a9cd4a7d02ef/model')
model2 = torch.load('/Users/macbook/works/mlflow-example/mlruns/0/021906e767b44b87b915a9cd4a7d02ef/artifacts/model/data/model.pth')
print(model)
print(model2)
print("Test here")