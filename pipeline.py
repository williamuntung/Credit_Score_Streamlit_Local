from pathlib import Path
from data_ingestion import DataIngestion
from train import Training
from eval import Evaluate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

class ModelPipeline:
    """Pipeline"""
    def __init__(self, raw_data_path: str | Path, threshold = 0.7):
        """Define model dari hasil experiment di ipynb"""
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.metrics_threshold = threshold
        self.SEED = 42
        self.models = {
            "Logistic Regression (SAGA Optimizer)": LogisticRegression(
                C=1.0,
                max_iter=100,
                penalty="l2",
                solver="saga",
                random_state=self.SEED
            ),

            "Extremely Randomized Trees (Extra Trees)": ExtraTreesClassifier(
                max_depth=None,
                max_features=None,
                min_samples_split=10,
                random_state=self.SEED,
                class_weight="balanced"
            ),

            "Random Forest Ensemble": RandomForestClassifier(
                max_depth=None,
                min_samples_split=5,
                n_estimators=300,
                random_state=self.SEED,
                class_weight="balanced"
            ),

            "XGBoost (Extreme Gradient Boosting)": XGBClassifier(
                learning_rate=0.05,
                max_depth=6,
                n_estimators=500,
                subsample=0.8,
                random_state=self.SEED
            ),

            "LightGBM (Light Gradient Boosting)": LGBMClassifier(
                learning_rate=0.1,
                n_estimators=200,
                num_leaves=63,
                verbosity=-1,
                random_state=self.SEED
            ),

            "CatBoost (Categorical Boosting)": CatBoostClassifier(
                depth=8,
                iterations=1000,
                learning_rate=0.05,
                random_state=self.SEED,
                verbose=0
            ),

            "Histogram-based Gradient Boosting": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=31,
                random_state=self.SEED
            )
        }
        
        # Component
        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.trainer = Training()
        self.evaluator = Evaluate()
        
    def execute(self):
        """Run Pipeline"""
        start_time = datetime.now().replace(microsecond=0)
        print(f"Start Running Pipeline: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Running Pipeline")
        
        ingested_file_path = self.ingestor.run()
        
        run_ids, x_test, y_test = self.trainer.run(ingested_file_path, self.models)
        
        metrics = self.evaluator.run(run_ids, x_test, y_test)
        
        counter = {}
        
        for model_name, model_metrics in metrics.items():
            counter[model_name] = 0
            for metric, value in model_metrics.items():
                if value > self.metrics_threshold:
                    counter[model_name] += 1
        
        idx = 1
        for model_name, count in counter.items():
            if count > 2:
                print(f"{idx}. Model {model_name} Approve")
            else:
                print(f"{idx}. Model {model_name} Rejected")
            idx += 1
            
        print("Running Pipeline Finish")
        end_time = datetime.now().replace(microsecond=0)
        duration = end_time - start_time
        print(f"Finish Running Pipeline: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Running Pipeline: {duration}")
        
if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "data_A.csv"
    
    pipeline = ModelPipeline(raw_data_path=DATA_INPUT)
    pipeline.execute()