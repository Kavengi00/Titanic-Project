import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pyngrok import ngrok


# Set the title of the app
st.set_page_config(page_title='Predicting Survival of Titanic Passengers', 
                   initial_sidebar_state='auto', page_icon="🚢")

st.title("Predicting Survival of Titanic Passengers", "🚢")

# load the saved model
load_model = pickle.load(open('C:/Users/ADMIN/Desktop/ML_Project/ML_Project/ml_classification/finalized_model.sav', 'rb'))
# Load the saved  scaler
scaler =  pickle.load(open('C:/Users/ADMIN/Desktop/ML_Project/ML_Project/ml_classification/scaler.pkl', 'rb'))

def get_user_input():
    # Use Streamlit widgets to get user input
    pclass = st.sidebar.radio('Pclass', [1, 2, 3])
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 0, 100, 30)  # min, max, default values
    sibsp = st.sidebar.slider('SibSp', 0, 8, 0)  # min, max, default values
    fare = st.sidebar.slider('Fare', 0.0, 512.0, 32.0)  # min, max, default values
    

    # Create a data frame from the inputs
    user_data = {'Sex': sex, 'Fare': fare, 'Pclass': pclass, 'Age': age, 'SibSp': sibsp}
    features = pd.DataFrame(user_data, index=[0])
    
    # Manually encode 'Sex' column
    features['Sex'] = features['Sex'].map({'Male': 0, 'Female': 1})

    # Transform 'Fare' and 'Age' columns
    features[['Age', 'Fare']] = scaler.transform(features[['Age','Fare']])
    
    return features

print(scaler.feature_names_in_)

# Display an image
st.image('titanic_ship.webp', caption='The Titanic Ship', use_column_width=True)

def main():
    # Get user input
    data = get_user_input()
    # Set seaborn style and color palette
    sns.set_style('darkgrid')

    # Create a button for the feature importance visualization
    if st.button('Show Feature Importance'):
    
    # Assuming load_model is your trained model and data is your DataFrame
        if hasattr(load_model, 'feature_importances_'):
            importances = load_model.feature_importances_

            # Convert the importances into a pandas Series for easier handling
            importances_series = pd.Series(importances, index=data.columns)

            # Sort the importances
            sorted_importances = importances_series.sort_values(ascending=False)
            
            # Create a color palette with the same number of colors as features
            palette = sns.color_palette('Paired', n_colors=len(sorted_importances))

            # Bar chart of feature importances
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x=sorted_importances.index, y=sorted_importances.values, ax=ax, palette=palette)
            ax.set_title('Feature Importances')
            plt.xlabel('Features')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
    # Predict button
    if st.button("Predict"):
        # Make prediction
        result = load_model.predict(data)
        proba = load_model.predict_proba(data)

        # Display result
        if result[0] == 1:
            st.write("***Congratulations !!!....*** **You probably would have made it!**")
            st.image("lifeboat.jpg")
            st.write(f"**Survival Probability Chances :** 'NO': {round((proba[0,0])*100,2)}%  'YES': {round((proba[0,1])*100,2)}%")
        else:
            st.write("***Better Luck Next time !!!!...*** **You're probably ended up like 'Jack'**")
            st.image("r.i.p.jpg")
            st.write(f"**Survival Probability Chances :** 'NO': {round((proba[0,0])*100,2)}%  'YES': {round((proba[0,1])*100,2)}%")
            
    #if st.button("Author"):
        #st.write("## @ Kavengi00")
        #st.write("Kavengi is graduatestudent with a passion for data science and machine learning.")
        #st.write("You can find more of his work on [GitHub](https://github.com/abhijit).")
if __name__ == "__main__":
    main()