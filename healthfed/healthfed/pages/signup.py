import streamlit as st
import pandas as pd
import os
import time

st.set_page_config(page_title="Signup", layout="centered")

# =========================
# ADVANCED UI STYLE
# =========================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#f6f0ff,#ffffff);
font-family:Segoe UI;
}

/* Sidebar style */

[data-testid="stSidebar"]{
background:linear-gradient(180deg,#efe7ff,#ffffff);
border-right:1px solid #e6dbff;
}

[data-testid="stSidebar"] *{
color:#4b3fa3;
}

/* Buttons */

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

/* Titles */

.page-title{
text-align:center;
color:#5e4aa8;
font-size:34px;
font-weight:700;
}

.page-subtitle{
text-align:center;
color:#666;
margin-bottom:30px;
}

/* Footer */

.footer{
text-align:center;
color:#777;
font-size:13px;
margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🩺 Health AI Panel")
    st.write("AI-Powered Medical Prediction")

    st.markdown("---")

    st.info("Create an account to access the healthcare AI system.")

    st.markdown("---")

    st.success("System Status: Online")

    st.markdown("### Platform Features")

    st.write("""
• Multi-Disease Prediction  
• Explainable AI (SHAP + LIME)  
• Patient Risk Analysis  
• Medical Insights
""")

# =========================
# TITLE
# =========================

st.markdown("<div class='page-title'>📝 Create Account</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Join the Federated Healthcare AI Platform</div>", unsafe_allow_html=True)

# =========================
# CENTERED SIGNUP FORM
# =========================

col1, col2, col3 = st.columns([1,2,1])

with col2:

    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")

    signup = st.button("Signup", use_container_width=True)

# =========================
# SIGNUP LOGIC
# =========================

if signup:

    if username == "" or password == "":
        st.warning("Please enter username and password")
        st.stop()

    if os.path.exists("users.csv"):
        users = pd.read_csv("users.csv")
    else:
        users = pd.DataFrame(columns=["username","password"])

    if username in users["username"].values:
        st.error("Username already exists")

    else:

        new_user = pd.DataFrame({
            "username":[username],
            "password":[password]
        })

        users = pd.concat([users,new_user], ignore_index=True)
        users.to_csv("users.csv", index=False)

        st.success("Signup successful")
        st.info("Redirecting to login...")

        time.sleep(2)

        st.switch_page("pages/1_Login.py")

# =========================
# FOOTER
# =========================

st.markdown(
"<div class='footer'>Federated Healthcare AI System • Secure Medical AI Platform</div>",
unsafe_allow_html=True
)