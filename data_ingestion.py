from pathlib import Path
import pandas as pd
import numpy as np

class DataIngestion:
    """Load, Cleaning, Saving Data"""
    
    # setup untuk cleaning
    
    # Irrelevant Data
    DROP_COLS = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN']
    
    CAT_TO_NUM = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Amount_invested_monthly', 'Monthly_Balance','Changed_Credit_Limit', 'Outstanding_Debt']

    # Set data non negative
    NON_NEGATIVE_COLS = [
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
        'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries',
        'Credit_History_Age', 'Annual_Income', 'Outstanding_Debt',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
    ]
 
    # kolom yang akan di handle outliernya
    OUTLIER_COLS = [
        'Num_Credit_Inquiries', 'Interest_Rate', 'Num_Credit_Card',
        'Num_Bank_Accounts', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Annual_Income', 'Total_EMI_per_month', 'Amount_invested_monthly'
    ]
 
    # cast to INT, karena outlier di ganti dengan np.nan maka data type berubah menjadi float
    FINAL_INT_COLS = [
        'Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Delay_from_due_date',
        'Num_of_Loan', 'Interest_Rate', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries',
    ]
 
    # inconsistance data kategorikal
    CATEGORICAL_PLACEHOLDERS = {
        'Occupation': '_______',
        'Credit_Mix': '_',
        'Payment_of_Min_Amount': 'NM',
        'Payment_Behaviour': '!@9#%8',
    }
    
    def __init__(self, input_path: str | Path, output_dir: str | Path):
        self.input_file = Path(input_path)
        self.ingested_dir = Path(output_dir)
        self.output_file = self.ingested_dir / "data_A.csv"
        
        self._exist_cols = sorted(set(
            self.DROP_COLS + self.CAT_TO_NUM
            + self.NON_NEGATIVE_COLS + self.OUTLIER_COLS + list(self.CATEGORICAL_PLACEHOLDERS.keys())
            + ['Credit_History_Age', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Type_of_Loan']
        ))
    
    def run(self) -> Path:
        """Run Data Ingestion"""
        print("-"*4, "1. Data Ingestion", "-"*4)
        
        self.ingested_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_csv(self.input_file)
        
        assert not df.empty, "Dataset Kosong!"
        
        df = self._clean_data(df)
        
        df.to_csv(self.output_file, index = False)
        print(f"Data Ingestion berhasil")
        print(f"{self.input_file} -> {self.output_file}")
        
        return self.output_file
    
    # -------------------------- #
    # Private: pipeline cleaning #
    # -------------------------- #
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data Cleaning"""
        self._validate_columns(df)
        
        df = self._drop_unused_columns(df)
        df = self._remove_underscore(df)
        df = self._parse_credit_history_age(df)
        df = self._clean_categorical_placeholders(df)
        df = self._clean_invalid_ranges(df)
        df = self._cut_outliers_iqr(df)
        df = self._clean_anomaly(df)
        df = self._cast_final_int_columns(df)
        df = self._encode_type_of_loan(df)
        df = self._encode_payment_behavior(df)
        df = self._adjust_num_of_loan(df)
        df = self._rename_cols(df)
        
        return df
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validasi kolom dari data mentah"""
        missing = [col for col in self._exist_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Kolom berikut tidak ditemukan di raw data: {missing}")
        
    def _drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hapus kolom yang tidak relevan"""
        return df.drop(columns=self.DROP_COLS)

    def _remove_underscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hapus Underscore yang tidak relevan untuk data numerik yang bertipe object"""
        for col in self.CAT_TO_NUM:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('_', '', regex=False)
                .pipe(pd.to_numeric, errors='coerce')
            )

        return df

    def _parse_credit_history_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ubah format teks 'X Years and Y Months' menjadi total bulan (Int64)"""
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(
            lambda x: int(x.split(" Years")[0]) * 12 + int(x.split("and")[1].split("Months")[0])
            if pd.notna(x) else np.nan
        )
        df['Credit_History_Age'] = df['Credit_History_Age'].astype('Int64')
        return df

    def _clean_categorical_placeholders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merubah CATEGORICAL_PLACEHOLDER menjadi NAN"""
        for col, placeholder in self.CATEGORICAL_PLACEHOLDERS.items():
            df[col] = df[col].replace(placeholder, np.nan)
        return df

    def _clean_invalid_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set ke NaN untuk nilai yang secara logis tidak valid (di luar rentang wajar)"""
        # Age: usia di luar rentang wajar dianggap invalid
        df.loc[(df['Age'] < 18) | (df['Age'] >= 100), 'Age'] = np.nan
    
        # Kolom-kolom yang seharusnya tidak boleh negatif
        for col in self.NON_NEGATIVE_COLS:
            df.loc[df[col] < 0, col] = np.nan
    
        # Credit_Utilization_Ratio: harus berupa persentase 0-100
        df.loc[
            (df['Credit_Utilization_Ratio'] < 0) | (df['Credit_Utilization_Ratio'] > 100),
            'Credit_Utilization_Ratio'
        ] = np.nan
    
        return df

    def _cut_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Potong outlier pada kolom tertentu"""
        for col in self.OUTLIER_COLS:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
    
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
    
            df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
    
        return df

    def _clean_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anggap invalid jika ada anomali EMI / Investasi"""
        df.loc[df['Total_EMI_per_month'] * 12 > df['Annual_Income'], 'Total_EMI_per_month'] = np.nan
        df.loc[df['Amount_invested_monthly'] > df['Annual_Income'] / 12, 'Amount_invested_monthly'] = np.nan
        return df

    def _cast_final_int_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ubah kolom integer kedalam Int64 karena sebelumnya ada NAN otomatis jadi float"""
        for col in self.FINAL_INT_COLS:
            df[col] = df[col].astype('Int64')
        return df

    def _encode_type_of_loan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Membuat kolom dummy dari kolom tipe pinjaman nasabah"""
        df['Type_of_Loan'] = df['Type_of_Loan'].str.replace('and ', '', regex=False)
    
        loan_dummies = df['Type_of_Loan'].str.get_dummies(sep=', ')
        df = pd.concat([df, loan_dummies], axis=1)
    
        return df

    def _encode_payment_behavior(self, df:pd.DataFrame) -> pd.DataFrame:
        """Extract 2 kolom dari string X_spent_Y_value_payments | X - Spending_Level | Y - Payment_Value_Level"""
        df[['Spending_Level', 'Payments_Value_Level']] = (
            df['Payment_Behaviour']
            .str.replace('_spent_', '_')
            .str.replace('_value_payments', '')
            .str.split('_', expand=True)
        )
        
        df = df.drop(columns='Payment_Behaviour')
        
        return df

    def _adjust_num_of_loan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Input NAN pada kolom jumlah pinjaman sesuai dengan jumlah tipe pinjaman"""
        
        df['Type_of_Loan_Count'] = df['Type_of_Loan'].str.count(', ') + 1
        
        df['Num_of_Loan'] = df['Num_of_Loan'].fillna(df['Type_of_Loan_Count'])
        
        df = df.drop(columns='Type_of_Loan')
        df = df.drop(columns='Type_of_Loan_Count')
        
        return df
        

    def _rename_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merubah nama kolom agar konsisten"""
        
        mapping = {
            'Auto Loan': 'Auto_Loan',
            'Credit-Builder Loan': 'Credit_Builder_Loan',
            'Debt Consolidation Loan': 'Debt_Consolidation_Loan',
            'Home Equity Loan': 'Home_Equity_Loan',
            'Mortgage Loan': 'Mortgage_Loan',
            'Not Specified': 'Not_Specified',
            'Payday Loan': 'Payday_Loan',
            'Personal Loan': 'Personal_Loan',
            'Student Loan': 'Student_Loan'
        }
        df = df.rename(columns = mapping)
        
        return df
    
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    RAW_DIR = BASE_DIR  
    INGESTED_DIR = BASE_DIR / "ingested"  
 
    ingestion = DataIngestion(
        input_path=RAW_DIR / "data_A.csv",
        output_dir=INGESTED_DIR,
    )
    ingestion.run()