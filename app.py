import streamlit as st
import joblib
import pandas as pd

# Load the machine learning model and encode
@st.cache_resource
def load_model():
    return joblib.load('artifacts/Extremely Randomized Trees (Extra Trees).pkl')

model = load_model()

LOAN_MAPPING = {'Yes': 1, 'No': 0}

LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
    "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan",
]

def main():
    st.title('Credit Score Prediction - Model Deployment')
    
    # User Input (sidebar, slider, radio, multi-select, number_input, ...)
    with st.sidebar:
        st.header("Data Nasabah")

        Month = st.radio(
            "Bulan pengambilan atau pencatatan data nasabah",
            ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
        )
        Age = st.slider("Usia nasabah (dalam satuan tahun)", 18, 100, step=1)
        Occupation = st.selectbox(
            "Pekerjaan atau profesi utama nasabah",
            ['Developer', 'Musician', 'Scientist', 'Entrepreneur', 'Accountant', 'Journalist',
             'Media_Manager', 'Mechanic', 'Writer', 'Doctor', 'Teacher', 'Lawyer', 'Engineer',
             'Architect', 'Manager']
        )
        Annual_Income = st.number_input(
            "Total pendapatan kotor nasabah dalam satu tahun", 0.0, 25000000.0, step=0.001
        )
        Monthly_Inhand_Salary = st.number_input(
            "Gaji bersih nasabah", 100.0, 20000.0, step=0.000001
        )
        Num_Bank_Accounts = st.slider("Jumlah rekening bank aktif nasabah", 0, 15, step=1)
        Num_Credit_Card = st.slider("Jumlah kartu kredit aktif nasabah", 0, 15, step=1)
        Interest_Rate = st.slider("Suku bunga rata-rata pada pinjaman atau kartu kredit nasabah", 0, 40, step=1)
        Num_of_Loan = st.slider("Jumlah total pinjaman berjalan nasabah", 0, 10, step=1)
        Delay_from_due_date = st.slider("Jumlah hari keterlambatan nasabah dari tanggal jatuh tempo", 0, 100, step=1)
        Num_of_Delayed_Payment = st.slider("Frekuensi atau berapa kali nasabah terlambat membayar tagihan dalam periode tertentu", 0, 50, step=1)
        Changed_Credit_Limit = st.number_input(
            "Persentase atau nominal perubahan limit (batas maksimal) kartu kredit nasabah "
            "sejak pengisian data terakhir",
            -50.0, 50.0, step=0.000001
        )
        Num_Credit_Inquiries = st.slider(
            "Jumlah pengecekan riwayat kredit nasabah oleh lembaga keuangan", 0, 20, step=1
        )
        Credit_Mix = st.radio(
            "Kombinasi atau keragaman jenis kredit yang dimiliki nasabah",
            ['Bad', 'Standard', 'Good']
        )
        Outstanding_Debt = st.number_input(
            "Total sisa utang keseluruhan yang belum dilunasi oleh nasabah", 0.0, 5000.0, step=0.000001
        )
        Credit_Utilization_Ratio = st.slider(
            "Persentase penggunaan limit kredit nasabah dibandingkan total limit yang tersedia",
            10.0, 50.0, step=0.000001
        )
        Credit_History_Age = st.slider(
            'Umur riwayat kredit nasabah (dalam bulan)', 0, 1000, step=1
        )
        Payment_of_Min_Amount = st.radio(
            "Indikator apakah nasabah sering membayar tagihan kartu kredit hanya dalam jumlah minimum saja",
            ['Yes', 'No']
        )
        Total_EMI_per_month = st.number_input(
            "Total cicilan bulanan nasabah", 0.0, 90000.0, step=0.000001
        )
        Amount_invested_monthly = st.number_input(
            "Jumlah uang yang diinvestasikan oleh nasabah setiap bulan", 0.0, 2000.0, step=0.000001
        )
        Monthly_Balance = st.number_input(
            "Sisa uang atau saldo yang dimiliki nasabah di akhir bulan", 0.0, 2000.0, step=0.000001
        )
        Spending_Level = st.radio('Pola spending nasabah dalam level', ['Small', 'Medium', 'Large'])
        Payments_Value_Level = st.radio("Level besar uang yang di spending nasabah", ['Low', 'High'])
 
        # Jenis-jenis pinjaman yang dimiliki nasabah (one-hot, masing-masing Yes/No)
        selected_loans = st.multiselect(
            "Jenis pinjaman apa saja yang dimiliki nasabah?", LOAN_TYPES
        )
        Auto_Loan = "Yes" if "Auto Loan" in selected_loans else "No"
        Credit_Builder_Loan = "Yes" if "Credit-Builder Loan" in selected_loans else "No"
        Debt_Consolidation_Loan = "Yes" if "Debt Consolidation Loan" in selected_loans else "No"
        Home_Equity_Loan = "Yes" if "Home Equity Loan" in selected_loans else "No"
        Mortgage_Loan = "Yes" if "Mortgage Loan" in selected_loans else "No"
        Not_Specified = "Yes" if "Not Specified" in selected_loans else "No"
        Payday_Loan = "Yes" if "Payday Loan" in selected_loans else "No"
        Personal_Loan = "Yes" if "Personal Loan" in selected_loans else "No"
        Student_Loan = "Yes" if "Student Loan" in selected_loans else "No"
 
        # Mapping jawaban Yes/No menjadi 1/0
        Auto_Loan = LOAN_MAPPING[Auto_Loan]
        Credit_Builder_Loan = LOAN_MAPPING[Credit_Builder_Loan]
        Debt_Consolidation_Loan = LOAN_MAPPING[Debt_Consolidation_Loan]
        Home_Equity_Loan = LOAN_MAPPING[Home_Equity_Loan]
        Mortgage_Loan = LOAN_MAPPING[Mortgage_Loan]
        Not_Specified = LOAN_MAPPING[Not_Specified]
        Payday_Loan = LOAN_MAPPING[Payday_Loan]
        Personal_Loan = LOAN_MAPPING[Personal_Loan]
        Student_Loan = LOAN_MAPPING[Student_Loan]
    
    data = {
        "Month": Month,
        "Age": int(Age),
        "Occupation": Occupation,
        "Annual_Income": Annual_Income,
        "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
        "Num_Bank_Accounts": int(Num_Bank_Accounts),
        "Num_Credit_Card": int(Num_Credit_Card),
        "Interest_Rate": int(Interest_Rate),
        "Num_of_Loan": int(Num_of_Loan),
        "Delay_from_due_date": int(Delay_from_due_date),
        "Num_of_Delayed_Payment": int(Num_of_Delayed_Payment),
        "Changed_Credit_Limit": Changed_Credit_Limit,
        "Num_Credit_Inquiries": int(Num_Credit_Inquiries),
        "Credit_Mix": Credit_Mix,
        "Outstanding_Debt": Outstanding_Debt,
        "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
        "Credit_History_Age": int(Credit_History_Age),
        "Payment_of_Min_Amount": Payment_of_Min_Amount,
        "Total_EMI_per_month": Total_EMI_per_month,
        "Amount_invested_monthly": Amount_invested_monthly,
        "Monthly_Balance": Monthly_Balance,
        "Auto_Loan": Auto_Loan,
        "Credit_Builder_Loan": Credit_Builder_Loan,
        "Debt_Consolidation_Loan": Debt_Consolidation_Loan,
        "Home_Equity_Loan": Home_Equity_Loan,
        "Mortgage_Loan": Mortgage_Loan,
        "Not_Specified": Not_Specified,
        "Payday_Loan": Payday_Loan,
        "Personal_Loan": Personal_Loan,
        "Student_Loan": Student_Loan,
        "Spending_Level": Spending_Level,
        "Payments_Value_Level": Payments_Value_Level
    }
    
    df=pd.DataFrame([data], columns=['Month','Age','Occupation','Annual_Income','Monthly_Inhand_Salary', 
                                     'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan', 
                                     'Delay_from_due_date','Num_of_Delayed_Payment','Changed_Credit_Limit',
                                     'Num_Credit_Inquiries','Credit_Mix','Outstanding_Debt','Credit_Utilization_Ratio',
                                     'Credit_History_Age','Payment_of_Min_Amount','Total_EMI_per_month',
                                     'Amount_invested_monthly','Monthly_Balance','Auto_Loan','Credit_Builder_Loan',
                                     'Debt_Consolidation_Loan','Home_Equity_Loan','Mortgage_Loan','Not_Specified',
                                     'Payday_Loan', 'Personal_Loan','Student_Loan','Spending_Level','Payments_Value_Level'])
    
    if st.button("Make Prediction"):
        prediction = model.predict(df)[0]
        target_mapping = {
            0: "Poor",
            1: "Standard",
            2: "Good"
        }
        
        answer = target_mapping[prediction]
        
        st.success(f"Credit Score Prediction: {answer}")

        # Grafik confidence model (probabilitas per kelas), jika model mendukung predict_proba
        probabilities = model.predict_proba(df)[0]
        classes = model.classes_

        classes_label = [target_mapping[c] for c in classes]

        proba_df = pd.DataFrame({
            "Kelas": classes_label,
            "Probabilitas": probabilities
        }).sort_values(
            "Probabilitas",
            ascending=False
        ).set_index("Kelas")

        st.subheader("Confidence Model")
        st.bar_chart(proba_df)


if __name__ == "__main__":
    main()