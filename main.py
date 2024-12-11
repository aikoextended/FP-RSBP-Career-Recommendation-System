import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_prepare_model():
    # Load the dataset
    dataset = pd.read_csv(r'C:\Users\User_\OneDrive\Documents\RSBP\sample_data\dataset9000.csv')

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
    k = 5
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

    # Create input fields
    st.header('Enter Your Skills Level')
    user_input = {}
    skill_levels = ['Not Interested', 'Beginner', 'Average', 'Intermediate', 'Excellent', 'Professional']

    for column in X.columns:
        user_input[column] = st.selectbox(
            f'Select your level for {column}',
            skill_levels,
            key=column
        )

    # Predict button
    if st.button('Get Career Recommendations'):
        try:
            # Get top role predictions
            top_roles, top_probabilities = predict_top_roles(model, label_encoders, X, user_input)

            # Display recommendations
            st.header('Top Career Recommendations')
            for role, prob in zip(top_roles, top_probabilities):
                st.metric(label=role, value=f'{prob * 100:.2f}%')

        except Exception as e:
            st.error(f'An error occurred: {e}')


if __name__ == '__main__':
    main()
