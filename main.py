import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_prepare_model():
    # Load the dataset
    dataset = pd.read_csv(r'C:\Users\Lenovo\Documents\FP-RSBP\sample_data\dataset9000.csv')

    # Handle missing values
    dataset['Graphics Designing'].fillna('Not Interested', inplace=True)

    # Encode categorical features
    label_encoders = {}
    for column in dataset.columns:
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        label_encoders[column] = le

    # Separate features (X) and target (y)
    X = dataset.drop(columns=['Role'])
    y = dataset['Role']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train K-Nearest Neighbors Classifier
    k = 3
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Calculate model accuracy
    y_pred = model.predict(X_test)
    accuracy = np.round(np.mean(y_pred == y_test) * 100, 2)

    return model, label_encoders, X, accuracy

def predict_top_roles(model, label_encoders, X, user_input):
    # Encode input
    encoded_input = {}
    for column in X.columns:
        encoded_input[column] = label_encoders[column].transform([user_input[column]])[0]

    # Convert input to DataFrame
    user_data = pd.DataFrame([encoded_input])

    # Predict probabilities
    probabilities = model.predict_proba(user_data)[0]

    # Get top 3 predictions
    top_indices = probabilities.argsort()[-3:][::-1]
    top_roles = label_encoders['Role'].inverse_transform(top_indices)
    top_probabilities = probabilities[top_indices]

    return top_roles, top_probabilities

def main():
    st.title('Career Recommendation System')

    # Load model and prepare data
    model, label_encoders, X, accuracy = load_and_prepare_model()

    # Display model accuracy
    st.sidebar.info(f'Model Accuracy: {accuracy}%')

    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state['page'] = 'start'
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = {}

    # Start page
    if st.session_state['page'] == 'start':
        if st.button('Start'):
            st.session_state['page'] = 'form'

    # Form page
    if st.session_state['page'] == 'form':
        st.header('Enter Your Skills Level')
        skill_levels = ['Not Interested', 'Poor', 'Beginner', 'Average', 'Intermediate', 'Excellent', 'Professional']

        for column in X.columns:
            st.session_state['user_input'][column] = st.selectbox(
                f'Select your level for {column}',
                skill_levels,
                key=column
            )

        if st.button('Get Career Recommendations'):
            st.session_state['page'] = 'results'

    # Results page
    if st.session_state['page'] == 'results':
        st.header('Top Career Recommendations')
        try:
            user_input = st.session_state['user_input']
            top_roles, top_probabilities = predict_top_roles(model, label_encoders, X, user_input)
            for role, prob in zip(top_roles, top_probabilities):
                st.progress(float(prob))
                st.text(f'{role}: {prob * 100:.2f}%')

        except Exception as e:
            st.error(f'An error occurred while displaying results: {e}')

        if st.button('Restart'):
            st.session_state['page'] = 'start'
            st.session_state['user_input'] = {}

if __name__ == '__main__':
    main()
