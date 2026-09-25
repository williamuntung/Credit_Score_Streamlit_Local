from typing import Dict
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluate:
    """Evaluasi Tiap model dengan load model dari mlflow"""
    def run(self, run_ID: Dict[str, str], x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Menerima input data dictionary {nama model, run_id}"""
        print("-"*4, "3. Evaluating", "-"*4)
        
        results = {}
        
        for model_name, run_id in run_ID.items():
            results[model_name] = self._evaluate_model(model_name, run_id, x_test, y_test)
        
        return results
    
    def _evaluate_model(self, model_name: str, run_ID: str, x_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluasi Models"""
        model_uri = f"runs:/{run_ID}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        preds = model.predict(x_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average = 'weighted'),
            "recall": recall_score(y_test, preds, average = "weighted"),
            "f1": f1_score(y_test, preds, average = "weighted")
        }
        
        with mlflow.start_run(run_id=run_ID):
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
                
        print(
            f"{model_name}\nAccuracy = {metrics['accuracy']:.3f}\nPrecision = {metrics['precision']:.3f}\nRecall = {metrics['recall']:.3f}\nF1 = {metrics['f1']:.3f}\n"
        )
        
        return metrics