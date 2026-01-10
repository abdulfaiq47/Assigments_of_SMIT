import streamlit as st 
import joblib # to laod pickle file import joblib
import numpy as np  # for arranging all input in one 


model = joblib.load("House_pred.pkl")  # loaded


st.title("House Prediction App") # For the title of web APP


st.write("Welcome to house prediction Streamlit app made by Faiq.")    


st.markdown("""
<style>
[data-testid="stVerticalBlock"] {
    background: #05040a;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    area_st = st.number_input("Enter Area in sq-ft: ", min_value=0, step=1)   # get area     in Numbers  & min_value to avoid negative number and setp =  1 mean take step 1 then 2 then 3 not 0.00 then 0.01 then 0.02
    bedrooms_st = st.number_input("Enter bedrooms: ", min_value=0, step=1)    # get area     in Numbers
with col2:    
    bathrooms_st = st.number_input("Enter bathrooms: ", min_value=0, step=1)  # get bathroom in Numbers    
    stories_st = st.number_input("Enter stories: ", min_value=0, step=1)      # get Stories  in Numbers   
with col3:      
    parking_st = st.number_input("Enter parking: ", min_value=0, step=1)      # get parking  in Numbers   

st.markdown("###  House Features")

c1, c2, c3 = st.columns(3)

with c1:
    mainroad = st.selectbox(         # get select box of yes or know 
    "Do it have Main road?",     # for display
    ["Yes", "No"]                # display only yes or NO 
)

    mainroad_st = 1 if mainroad == "Yes" else 0  # get condition if yes then covert into 1 other wise 0
    st.write(mainroad_st)   # for display values

    guestroom = st.selectbox( 
    "Do it have guestroom?",
    ["Yes", "No"]
)
    guestroom_st = 1 if guestroom == "Yes" else 0
    st.write(guestroom_st)
with c2:
    basement = st.selectbox(
    "Do it have basement?",
    ["Yes", "No"]
)
    basement_st = 1 if basement == "Yes" else 0
    st.write(basement_st)

    hotwaterheating = st.selectbox(
    "Do it have  hotwaterheating?",
    ["Yes", "No"]
)
    hotwaterheating_st = 1 if hotwaterheating == "Yes" else 0
    st.write(hotwaterheating_st)
with col1:
    airconditioning = st.selectbox(
    "Do it have airconditioning ?",
    ["Yes", "No"]
)
    airconditioning_st = 1 if airconditioning == "Yes" else 0
    st.write(airconditioning_st)
with c3:
    prefarea = st.selectbox(
    "Do it have prefarea?",
    ["Yes", "No"]
)
    prefarea_st = 1 if prefarea == "Yes" else 0
    st.write(prefarea_st)

    furnishingstatus = st.selectbox(
    "Do it have furnishingstatus?",
    ["furnished", "semi-furnished", "unfurnished"]   # here i give three value 
)
    if furnishingstatus == "furnished":            # here i perfeorm condition testing 
        furnishingstatus_st = 2
    elif furnishingstatus == "semi-furnished":
        furnishingstatus_st = 1
    else:
        furnishingstatus_st = 0    

    st.write(furnishingstatus_st)



input_data = np.array([[area_st, bedrooms_st, bathrooms_st, stories_st, parking_st,    # here i take 2 dim array Or nested list to stor all values in "Input_data"
                        mainroad_st, guestroom_st, basement_st, hotwaterheating_st,
                        airconditioning_st, prefarea_st, furnishingstatus_st]])



if st.button("Predict Price"):
    p_price =  model.predict(input_data)
    st.success(f"Predicted Price of House is : {p_price[0]}")
