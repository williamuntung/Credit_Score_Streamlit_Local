from pathlib import Path
from typing import Dict, Tuple
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
import os

class Preprocessor:
    """Processor yang digunakan berisi split dan transformer"""
    def __init__(self):
        self.ORD_COLS = ['Payments_Value_Level', 'Spending_Level', 'Payment_of_Min_Amount', 'Credit_Mix']
        self.OHE_COLS = ['Month', 'Occupation']
        self.INT_COLS = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
                    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries',
                    'Credit_History_Age']
        self.BIN_COLS = ['Auto_Loan', 'Credit_Builder_Loan', 'Debt_Consolidation_Loan', 'Home_Equity_Loan',
                    'Mortgage_Loan', 'Not_Specified', 'Payday_Loan', 'Personal_Loan', 'Student_Loan']
        self.FLT_COLS = ['Total_EMI_per_month', 'Changed_Credit_Limit', 'Credit_Utilization_Ratio',
                    'Monthly_Balance', 'Monthly_Inhand_Salary', 'Amount_invested_monthly',
                    'Annual_Income', 'Outstanding_Debt']
        
        # Kategori ordinal harus urut dari rendah ke tinggi, sejajar dengan ORD_COLS
        self.ORD_CATEGORIES = [
            ['Small', 'Medium', 'Large'], 
            ['Low', 'High'], 
            ['No', 'Yes'],
            ['Bad', 'Standard', 'Good'], 
        ]
    
    def split(self, data_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Spliting Data"""
        df = pd.read_csv(Path(data_path), sep=",")

        x = df.drop("Credit_Score", axis=1)
        y = df["Credit_Score"]

        return train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
    
    def transformer(self) -> ColumnTransformer:
        """Define Transformer"""
        int_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        flt_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        ohe_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore'
            ))
        ])

        ord_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=self.ORD_CATEGORIES, handle_unknown='use_encoded_value', unknown_value=-1))
        ])


        return ColumnTransformer([
            ('int', int_pipeline, self.INT_COLS),
            ('float', flt_pipeline, self.FLT_COLS),
            ('bin', 'passthrough', self.BIN_COLS),
            ('ohe', ohe_pipeline, self.OHE_COLS),
            ('ord', ord_pipeline, self.ORD_COLS)
        ], remainder='drop')
        
class Training:
    """Train lebih dari satu model dan simpan di mlflow"""
    _LOGGABLE_TYPES = (str, int, float, bool, type(None))
    
    def __init__(self, experiment_name: str = "Credit Score Classification", artifact_dir: str = "artifacts"):
        self.experiment_name = experiment_name
        self.artifacts_dir = Path(artifact_dir)
        self.preprocessor = Preprocessor()
        self.target_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
        
        os.makedirs(self.artifacts_dir, exist_ok=True)
        mlflow.set_experiment(self.experiment_name)
        
    def run(self, data_path: str | Path, models: Dict[str, BaseEstimator]) -> Dict[str, str]:
        """Train model menerima data dictionary {nama model: parameters}"""
        print("-"*4, "2. Preprocessing + Training", "-"*4)
        
        x_train, x_test, y_train, y_test = self.preprocessor.split(data_path)

        y_train = y_train.map(self.target_mapping)
        y_test = y_test.map(self.target_mapping)
        
        run_ids = {}
        
        for model_name, model in models.items():
            run_ids[model_name] = self._train_model(model_name, model, x_train, y_train)
            
        return run_ids, x_test, y_test
        
    def _train_model(self, model_name: str, model: BaseEstimator, x_train: pd.DataFrame, y_train: pd.Series) -> str:
        """Train model"""
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor.transformer()),
            ('classifier', model),
        ])
        
        with mlflow.start_run(run_name = model_name) as run:
            self._log_params(model)
            
            pipeline.fit(x_train, y_train)
            
            model_file_path = self.artifacts_dir / f"{model_name}.pkl"
            joblib.dump(pipeline, model_file_path)
        
            mlflow.sklearn.log_model(
                pipeline, 
                artifact_path="model",
                # registered_model_name=f"Credit_Score_Model_{model_name}",
                input_example=x_train.head()
            )
            
        print(f"{model_name} berhasil di train! | run_id = {run.info.run_id}")
        
        return run.info.run_id
    
    def _log_params(self, model: BaseEstimator) -> None:
        """Logging"""
        params = model.get_params()
        
        loggable_params = {
            key: (value if isinstance(value, self._LOGGABLE_TYPES) else str(value))
            for key, value in params.items()
        }
        
        mlflow.log_param("model_clf", type(model).__name__)
        for param, value in loggable_params.items():
            mlflow.log_param(param, value)