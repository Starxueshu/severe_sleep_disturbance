# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("A machine learning-based model to predict severe sleep disturbance among university students after analyzing life style, sport habit, and psychological health")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
drinkingperweek = st.sidebar.selectbox("Drinking frequency per week", ("0", "1", "2-3", "4-5", "≧6"))
Monthlyexpense = st.sidebar.selectbox("Monthly expense (￥)", ("0-1999", "2000-4999", "5000-9999", "≧10000"))
barbecue = st.sidebar.selectbox("Prefer eating barbecue", ("No", "Yes"))
Chronicdisease = st.sidebar.selectbox("Chronic disease", ("No", "Yes"))
GAD_7fj = st.sidebar.selectbox("Severity of anxiety", ("None", "Mild", "Moderate", "Severe"))
PHQ_9fj = st.sidebar.selectbox("Severity of depression", ("None", "Mild", "Moderate", "Moderate-to-severe", "Severe."))
DASSstressfj = st.sidebar.selectbox("Severity of stress", ("None", "Mild", "Moderate", "Severe", "Extremely severe"))
historyofsleepdisorder = st.sidebar.selectbox("History of sleep disorder", ("No", "Yes"))
historyofmentaldistress = st.sidebar.selectbox("History of mental distress", ("No", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[drinkingperweek, Monthlyexpense, barbecue, Chronicdisease, GAD_7fj, PHQ_9fj, DASSstressfj, historyofsleepdisorder, historyofmentaldistress]],
                     columns=["drinkingperweek", "Monthlyexpense", "barbecue", "Chronicdisease", "GAD_7fj", "PHQ_9fj", "DASSstressfj", "historyofsleepdisorder", "historyofmentaldistress"])
    x = x.replace(["0", "1", "2-3", "4-5", "≧6"], [0, 1, 2, 3, 4])
    x = x.replace(["0-1999", "2000-4999", "5000-9999", "≧10000"], [1, 2, 3, 4])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["None", "Mild", "Moderate", "Severe"], [1, 2, 3, 4])
    x = x.replace(["None", "Mild", "Moderate", "Moderate-to-severe", "Severe."], [1, 2, 3, 4, 5])
    x = x.replace(["None", "Mild", "Moderate", "Severe", "Extremely severe"], [1, 2, 3, 4, 5])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of severe sleep disturbance: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.298:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.298:
        st.markdown(f"Recommendations: Low-risk individuals, on the other hand, should maintain their good sleep habits and avoid behaviors that could disrupt their sleep. This includes avoiding late-night screen time, reducing stress levels, and engaging in regular physical activity. They can also try to further optimize their sleep environment by investing in a comfortable mattress and pillows and using white noise or blackout curtains to block out any external noise or light. Seeking professional help from a healthcare provider or sleep specialist may be necessary if sleep problems persist despite these measures.")
    else:
        st.markdown(f"Recommendations: For high-risk individuals, it is important to establish a regular sleep routine and stick to it, even on weekends. This includes going to bed and waking up at the same time every day, avoiding naps during the day, and limiting caffeine and alcohol intake, especially in the evening. They should also create a sleep-conducive environment by keeping their bedroom quiet, dark, and cool, and minimizing distractions such as electronic devices. For those who have trouble falling asleep, relaxation techniques such as deep breathing or meditation may be helpful. Seeking professional help from a healthcare provider or sleep specialist may be necessary if sleep problems persist despite these measures.")

st.subheader('Model information')
st.markdown('The model was established using the XGBoosting machine algorithm with the area under the curve of 0.872 [95%CI: 0.848-0.896]. This online calculator is able to assess the risk of severe sleep disturbance especially among university students. It is freely available and should be used for research only.')