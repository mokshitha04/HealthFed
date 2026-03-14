import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="Federated Multi Disease Prediction", layout="wide")

# =========================
# GLOBAL STYLE
# =========================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#f6f0ff,#ffffff);
font-family:Segoe UI;
}

[data-testid="stSidebar"]{
background:linear-gradient(180deg,#efe7ff,#ffffff);
border-right:1px solid #e6dbff;
}

[data-testid="stSidebar"] *{
color:#4b3fa3;
}

h1{
color:#5e4aa8;
text-align:center;
font-weight:700;
}

h3{
color:#6c58c4;
}

div.stButton > button{
background:linear-gradient(135deg,#8b7af7,#6c5ce7);
color:white;
border-radius:12px;
padding:10px 30px;
font-size:16px;
border:none;
box-shadow:0px 4px 12px rgba(0,0,0,0.15);
transition:0.3s;
}

div.stButton > button:hover{
transform:scale(1.05);
}

.feature-card{
padding:25px;
border-radius:18px;
background:linear-gradient(145deg,#ffffff,#f6f2ff);
box-shadow:0px 8px 18px rgba(0,0,0,0.08);
text-align:center;
transition:transform 0.35s ease, box-shadow 0.35s ease;
cursor:pointer;
}

.feature-card:hover{
transform:scale(1.08);
box-shadow:0px 18px 40px rgba(0,0,0,0.2);
}

.result-card{
padding:30px;
border-radius:18px;
background:linear-gradient(135deg,#ede7ff,#f9f7ff);
box-shadow:0px 8px 20px rgba(0,0,0,0.1);
}

.metric-card{
padding:20px;
border-radius:14px;
background:white;
box-shadow:0px 6px 15px rgba(0,0,0,0.08);
text-align:center;
}

.footer{
text-align:center;
color:#777;
font-size:14px;
margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN CHECK
# =========================

if "logged_in" not in st.session_state or st.session_state.logged_in == False:

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.markdown("""
        <h1>🔒 Access Restricted</h1>
        <p style='text-align:center;font-size:18px;color:#666'>
        Please login first to access the Federated Healthcare AI System.
        </p>
        """, unsafe_allow_html=True)

        st.warning("Go to the Login page from the sidebar.")

    st.stop()

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("🩺 Health AI Panel")

    st.write("AI-Powered Medical Prediction")

    st.markdown("---")
    st.info("Select disease and enter patient details to generate prediction.")
    st.markdown("---")

    st.success("System Status: Online")

    st.markdown("### Quick Guide")
    st.write("""
    1. Select a disease  
    2. Enter patient parameters  
    3. Click **Predict**  
    4. Review AI explanations
    """)
# =========================
# TOP BAR
# =========================

top1, top2, top3 = st.columns([6,1,1])

with top3:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.switch_page("pages/Login.py")
# =========================
# TITLE
# =========================

st.markdown("""
<h1>🩺 Federated Multi-Disease Prediction System</h1>
<p style='text-align:center;color:#666;font-size:18px'>
AI-Powered Healthcare Risk Assessment Platform
</p>
""", unsafe_allow_html=True)

# =========================
# FEATURE CARDS
# =========================

col1,col2,col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
    🧠 <b>AI Prediction</b><br>
    Detect disease risk using machine learning models.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
    🔬 <b>Explainable AI</b><br>
    SHAP & LIME explain predictions transparently.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
    🏥 <b>Healthcare Insights</b><br>
    View symptoms and medical recommendations.
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================
# SELECT DISEASE
# =========================

disease = st.selectbox(
"Select Disease",
["Select Disease","Diabetes","Heart Disease","Lung Disease","Liver Disease"]
)

model=None
input_df=None
training_data=None

# =========================
# DIABETES
# =========================

if disease=="Diabetes":

    model=joblib.load("diabetes_model.pkl")
    training_data=pd.read_csv("data/diabetes.csv")

    st.subheader("Patient Information")

    Pregnancies=st.number_input("Pregnancies",0,20,1)
    Glucose=st.number_input("Glucose Level",50,300,120)
    BloodPressure=st.number_input("Blood Pressure",40,200,70)
    SkinThickness=st.number_input("Skin Thickness",0,100,20)
    Insulin=st.number_input("Insulin Level",0,900,80)
    BMI=st.number_input("BMI",10.0,70.0,25.0)
    DPF=st.number_input("Diabetes Pedigree Function",0.0,5.0,0.5)
    Age=st.number_input("Age",10,100,40)

    cols=["Pregnancies","Glucose","BloodPressure","SkinThickness",
          "Insulin","BMI","DiabetesPedigreeFunction","Age"]

    vals=[Pregnancies,Glucose,BloodPressure,SkinThickness,
          Insulin,BMI,DPF,Age]

    input_df=pd.DataFrame([vals],columns=cols)

# =========================
# HEART DISEASE
# =========================

elif disease=="Heart Disease":

    model=joblib.load("heart_model.pkl")
    training_data=pd.read_csv("data/heart.csv")

    st.subheader("Patient Information")

    age=st.number_input("Age",20,100,50)
    sex=st.selectbox("Sex",[0,1])
    cp=st.number_input("Chest Pain Type",0,3,1)
    trestbps=st.number_input("Resting Blood Pressure",80,200,120)
    chol=st.number_input("Cholesterol",100,600,200)
    fbs=st.selectbox("Fasting Blood Sugar",[0,1])
    restecg=st.number_input("Rest ECG",0,2,1)
    thalach=st.number_input("Max Heart Rate",60,220,150)
    exang=st.selectbox("Exercise Angina",[0,1])
    oldpeak=st.number_input("Oldpeak",0.0,10.0,1.0)
    slope=st.number_input("Slope",0,2,1)
    ca=st.number_input("CA",0,4,0)
    thal=st.number_input("Thal",0,3,2)

    cols=["age","sex","cp","trestbps","chol","fbs","restecg",
          "thalach","exang","oldpeak","slope","ca","thal"]

    vals=[age,sex,cp,trestbps,chol,fbs,restecg,
          thalach,exang,oldpeak,slope,ca,thal]

    input_df=pd.DataFrame([vals],columns=cols)

# =========================
# LUNG
# =========================

elif disease=="Lung Disease":

    model=joblib.load("lung_model.pkl")
    training_data=pd.read_csv("data/survey lung cancer.csv")

    st.subheader("Patient Information")

    gender=st.selectbox("Gender",["Female","Male"])
    GENDER_F=1 if gender=="Female" else 0
    GENDER_M=1 if gender=="Male" else 0

    AGE=st.number_input("Age",10,100,40)

    SMOKING=st.selectbox("Smoking",[0,1])
    YELLOW=st.selectbox("Yellow Fingers",[0,1])
    ANXIETY=st.selectbox("Anxiety",[0,1])
    PEER=st.selectbox("Peer Pressure",[0,1])
    CHRONIC=st.selectbox("Chronic Disease",[0,1])
    FATIGUE=st.selectbox("Fatigue",[0,1])
    ALLERGY=st.selectbox("Allergy",[0,1])
    WHEEZING=st.selectbox("Wheezing",[0,1])
    ALCOHOL=st.selectbox("Alcohol Consuming",[0,1])
    COUGH=st.selectbox("Coughing",[0,1])
    BREATH=st.selectbox("Shortness of Breath",[0,1])
    SWALLOW=st.selectbox("Swallowing Difficulty",[0,1])
    CHEST=st.selectbox("Chest Pain",[0,1])

    input_dict={
        "AGE":AGE,
        "SMOKING":SMOKING,
        "YELLOW_FINGERS":YELLOW,
        "ANXIETY":ANXIETY,
        "PEER_PRESSURE":PEER,
        "CHRONIC DISEASE":CHRONIC,
        "FATIGUE ":FATIGUE,
        "ALLERGY ":ALLERGY,
        "WHEEZING":WHEEZING,
        "ALCOHOL CONSUMING":ALCOHOL,
        "COUGHING":COUGH,
        "SHORTNESS OF BREATH":BREATH,
        "SWALLOWING DIFFICULTY":SWALLOW,
        "CHEST PAIN":CHEST,
        "GENDER_F":GENDER_F,
        "GENDER_M":GENDER_M
    }

    input_df=pd.DataFrame([input_dict])

# =========================
# LIVER
# =========================

elif disease=="Liver Disease":

    model=joblib.load("liver_model.pkl")
    training_data=pd.read_csv("data/liver.csv")

    st.subheader("Patient Information")

    gender=st.selectbox("Gender",["Male","Female"])
    Gender=1 if gender=="Male" else 0

    Age=st.number_input("Age",10,100,40)
    TB=st.number_input("Total Bilirubin",0.1,20.0,1.0)
    DB=st.number_input("Direct Bilirubin",0.0,10.0,0.3)
    Alk=st.number_input("Alkaline Phosphotase",50,3000,200)
    ALT=st.number_input("ALT",10,2000,30)
    AST=st.number_input("AST",10,2000,30)
    TP=st.number_input("Total Proteins",2.0,10.0,6.0)
    ALB=st.number_input("Albumin",1.0,6.0,3.0)
    AGR=st.number_input("Albumin/Globulin Ratio",0.0,3.0,1.0)

    cols=["Age","Gender","Total_Bilirubin","Direct_Bilirubin",
          "Alkaline_Phosphotase","Alamine_Aminotransferase",
          "Aspartate_Aminotransferase","Total_Protiens",
          "Albumin","Albumin_and_Globulin_Ratio"]

    vals=[Age,Gender,TB,DB,Alk,ALT,AST,TP,ALB,AGR]

    input_df=pd.DataFrame([vals],columns=cols)


# =========================
# PREDICTION
# =========================

if model is not None and input_df is not None:

    if st.button("Predict"):

        prediction=model.predict(input_df)[0]
        prob=model.predict_proba(input_df)[0]
        risk_percent=round(prob[1]*100,2)

        st.markdown("<div class='result-card'>",unsafe_allow_html=True)

        st.subheader("Prediction Result")

        if prediction==1:
            st.error(f"{disease} Detected")
        else:
            st.success(f"No {disease} Detected")

        st.write("Risk Probability:",risk_percent,"%")
        st.progress(int(risk_percent))

        st.markdown("</div>",unsafe_allow_html=True)

        # Risk indicator
        if risk_percent < 30:
            st.success("🟢 Low Risk Level")
        elif risk_percent < 70:
            st.warning("🟡 Moderate Risk Level")
        else:
            st.error("🔴 High Risk Level")

        # Metrics
        c1,c2,c3=st.columns(3)

        with c1:
            st.metric("Risk %",f"{risk_percent}%")
        with c2:
            st.metric("Prediction","Disease" if prediction==1 else "Healthy")
        with c3:
            st.metric("Model","Federated ML")

        # Patient input summary
        summary=pd.DataFrame({
            "Feature":input_df.columns,
            "Value":input_df.values[0]
        })

        st.write("### Patient Input Summary")
        st.dataframe(summary,use_container_width=True)

        st.info("⚕️ AI predictions are for educational purposes only.")

        # =========================
        # SHAP
        # =========================

        with st.expander("🔍 AI Explanation (SHAP)",expanded=True):

            try:
                X_train=training_data.drop(training_data.columns[-1],axis=1)
                X_train=X_train.apply(pd.to_numeric,errors="coerce").fillna(0)
                X_train=X_train.reindex(columns=input_df.columns,fill_value=0)

                explainer=shap.Explainer(model,X_train)
                shap_values=explainer(input_df)

                shap.plots.waterfall(shap_values[0],show=False)
                st.pyplot(plt.gcf())

            except:
                st.warning("SHAP explanation unavailable.")

        # =========================
        # LIME
        # =========================

        with st.expander("🧠 Local Explanation (LIME)"):

            try:
                lime_explainer=lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=X_train.columns,
                    class_names=["No Disease","Disease"],
                    mode="classification"
                )

                exp=lime_explainer.explain_instance(
                    input_df.values[0],
                    model.predict_proba
                )

                fig3=exp.as_pyplot_figure()
                st.pyplot(fig3)

            except:
                st.warning("LIME explanation unavailable.")
        # =========================
        # SYMPTOMS
        # =========================

        st.markdown("### 🧬 Possible Symptoms")

        if disease=="Heart Disease":
            st.write("Chest pain, shortness of breath, fatigue, irregular heartbeat")

        elif disease=="Diabetes":
            st.write("Frequent urination, excessive thirst, fatigue, blurred vision")

        elif disease=="Lung Disease":
            st.write("Persistent cough, wheezing, breathing difficulty, chest pain")

        elif disease=="Liver Disease":
            st.write("Jaundice, fatigue, abdominal swelling, nausea")

        # =========================
        # OUTCOME / MEDICAL ADVICE
        # =========================

        st.markdown("### 🏥 Medical Recommendation")

        if prediction==1:
            st.warning("Model indicates higher risk. Medical consultation is recommended.")
        else:
            st.success("Model indicates lower risk but regular health monitoring is advised.")
st.markdown("<hr style='border:2px solid #e6dbff;'>",unsafe_allow_html=True)

st.markdown("<div class='footer'>Federated Healthcare AI System • Explainable AI Medical Platform</div>",unsafe_allow_html=True)