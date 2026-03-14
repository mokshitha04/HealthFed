import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="Login", layout="centered")

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

.login-title{
text-align:center;
color:#5e4aa8;
font-size:34px;
font-weight:700;
}

.login-subtitle{
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
# SESSION
# =========================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🩺 Health AI Panel")
    st.write("AI-Powered Medical Prediction")

    st.markdown("---")

    st.info("Secure login required to access the healthcare AI system.")

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

st.markdown("<div class='login-title'>🔐 Secure Login</div>", unsafe_allow_html=True)
st.markdown("<div class='login-subtitle'>Federated Healthcare AI Platform</div>", unsafe_allow_html=True)


# =========================
# CENTERED LOGIN FORM
# =========================

col1, col2, col3 = st.columns([1,2,1])

with col2:

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    login = st.button("Login", use_container_width=True)


# =========================
# LOGIN LOGIC
# =========================

if login:

    try:
        users = pd.read_csv("users.csv")
    except:
        st.error("users.csv not found")
        st.stop()

    user = users[
        (users["username"] == username) &
        (users["password"] == password)
    ]

    if not user.empty:

        st.session_state.logged_in = True
        st.session_state.user = username

        st.success("Login successful")
        st.info("Opening system...")

        time.sleep(2)

        st.rerun()

    else:
        st.error("Invalid username or password")


# =========================
# FOOTER
# =========================

st.markdown(
"<div class='footer'>Federated Healthcare AI System • Secure Medical AI Platform</div>",
unsafe_allow_html=True
)