import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import subprocess
import os
from datetime import datetime

# Load the pre-trained OCR model
@st.cache(allow_output_mutation = True)
def load_ocr_model():
    return load_model("OCR CNN.h5")

# Load the Feedback CSV file
@st.cache(allow_output_mutation = True)
def load_feedback_data():
    return pd.read_csv("feedback.csv")

# Main app logic
def main():
    st.set_page_config(page_title="Nexus OCR", page_icon="📝", layout="wide" ,initial_sidebar_state = "collapsed")

    
    col1, col2, col3 = st.columns([0.825, 2, 1])
    with col2:
        image = Image.open("Logo.jpg")
        st.image(image, width=1000)

    st.markdown("---")
    st.title("Nexus OCR")

    st.write("""
    Welcome to Nexus OCR, your intelligent companion for character recognition!

    Nexus OCR employs cutting-edge machine learning to transform your handwritten characters into digital text. Here's how to harness its power:

    1. Inscribe a single character (0-9 or A-Z) on the canvas below.
    2. Summon Nexus OCR's insight by clicking 'Predict'.
    3. Provide feedback on the accuracy of the prediction.
    4. If the prediction isn't correct, you can input the right character to enhance Nexus OCR's abilities.
    5. Use 'New Glyph' to start a fresh prediction.

    Experience the fusion of human creativity and artificial intelligence with Nexus OCR!

    **Note:** This project was developed as part of our learning journey in machine learning and web application development. Your feedback and interactions help us improve and learn. Thank you for being part of our educational experience!
    """)

    st.markdown("---")

    model = load_ocr_model()
    df = load_feedback_data()

    # Define the list of labels for prediction
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    def preprocess_and_predict(image_array):
        img = cv2.resize(image_array, (28, 28), interpolation=cv2.INTER_AREA)
        image = img.reshape(-1, 28, 28, 1)
        predicted_label_index = np.argmax(model.predict(image))
        predicted_label = label_list[predicted_label_index]
        return predicted_label

    def is_canvas_empty(image_data):
        return image_data is None or np.all(image_data[:, :, -1] == 0)

    # Initialize session state variables
    if 'stage' not in st.session_state:
        st.session_state.stage = 'draw'
    if 'predicted_label' not in st.session_state:
        st.session_state.predicted_label = None
    if 'img_array' not in st.session_state:
        st.session_state.img_array = None
    if 'canvas_key' not in st.session_state:
        st.session_state.canvas_key = 0

    # Function to reset the app state
    def reset_app_state():
        st.session_state.stage = 'draw'
        st.session_state.predicted_label = None
        st.session_state.img_array = None
        st.session_state.canvas_key += 1

    # Options for the canvas
    st.sidebar.header("Canvas Options")
    stroke_width = st.sidebar.slider("Stroke Width", 1, 50, 15)
    stroke_color = st.sidebar.color_picker("Stroke Color", "#000000")
    bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

    # Center the canvas using CSS
    st.markdown("""
        <style>
        .canvas-container { display: flex; justify-content: center;}
        .stButton > button {
            display: block;
            margin: 0 auto;
        }
        [data-testid="stHorizontalBlock"] {
            align-items: center;
            justify-content: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create a centered container for the canvas
    st.markdown("<h3 style='text-align: center;'>Draw Here:</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.45, 2, 1])
    with col2:
        canvas_result = st_canvas(
            fill_color=bg_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=600,
            width=600,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

    if st.session_state.stage == 'draw':
        if st.button("Predict", key="predict_button"):
            if is_canvas_empty(canvas_result.image_data):
                st.warning("The canvas is empty. Please draw something before predicting.")
            else:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                img = img.convert('L')
                st.session_state.img_array = np.array(img)
                st.session_state.predicted_label = preprocess_and_predict(st.session_state.img_array)
                st.session_state.stage = 'feedback'
                st.experimental_rerun()

    elif st.session_state.stage == 'feedback':
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 24px; margin-bottom: 15px;'>
                Predicted Character: <strong>{st.session_state.predicted_label}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Center-align the selection by using columns
        col1, col2, col3 = st.columns([2.5, 2, 1])
        with col2:
            # Using selectbox with an empty first option
            is_correct = st.selectbox(
                "Is the prediction correct?",
                options=["Select an option", "Yes", "No"],
                key="correct_select"
            )
        
        def git_push_changes():
            try:
                # Configure git (only needed first time)
                subprocess.run(["git", "config", "--global", "user.email", "neelotpal.santra@gmail.com"])
                subprocess.run(["git", "config", "--global", "user.name", "NSANTRA"])
                
                # Add the changed file
                subprocess.run(["git", "add", "feedback.csv"])
                
                # Create commit with timestamp
                commit_message = f"Update feedback data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                subprocess.run(["git", "commit", "-m", commit_message])
                
                # Push changes
                subprocess.run(["git", "push"])
                
                return True
            except Exception as e:
                st.error(f"Error pushing to git: {str(e)}")
                return False

        if is_correct == "Yes":
            st.success("Great! The prediction was correct.")
            if st.button("Start Over", key="start_over_yes"):
                reset_app_state()
                st.experimental_rerun()
                
        elif is_correct == "No":
            feedback_label = st.text_input("Please provide the correct character (0-9 or A-Z):")
            
            if feedback_label:
                if st.button("Submit Feedback"):
                    correct_label = feedback_label.upper()
                    if correct_label and len(correct_label) == 1 and (correct_label in [str(i) for i in range(10)] or 'A' <= correct_label <= 'Z'):
                        # Resize the image to 28x28 and flatten it into 784 pixels
                        ext_var = cv2.resize(st.session_state.img_array, (28, 28), interpolation=cv2.INTER_AREA)
                        ext_var = ext_var.flatten().reshape(1, -1)
                        
                        # Convert to DataFrame with pixel column names pixel1 to pixel784
                        columns = [f"pixel{i}" for i in range(1, 785)]
                        ext_var_df = pd.DataFrame(ext_var, columns=columns)
                        
                        # Add the correct label to the "class" column
                        ext_var_df["class"] = label_list.index(correct_label if correct_label.isalpha() else int(correct_label))
                        
                        # Append feedback to the DataFrame and save to CSV
                        df = pd.concat([df, ext_var_df], ignore_index=True)
                        df.to_csv("feedback.csv", index=False)
                        
                        # Push changes to git
                        if git_push_changes():
                            st.success(f"Thank you for your feedback! You've entered: {correct_label} and the changes have been pushed to the repository.")
                        else:
                            st.warning(f"Feedback saved locally but couldn't push to repository. You've entered: {correct_label}")
                    else:
                        st.warning("Please enter a valid single character (0-9 or A-Z).")

            # Separate "Start Over" button for the "No" case
            if st.button("Start Over", key="start_over_no"):
                reset_app_state()
                st.experimental_rerun()

    # Add contact information at the bottom of the page
    st.markdown("---")
    st.header("Contact Us")
    st.write("""
    We'd love to hear from you! Here's how you can reach us:
             
    - **Developers:** 
        1. [Neelotpal Santra](mailto:neelotpal.santra@gmail.com)
        2. [Ronak Parmar](mailto:ronakparmar1234@gmail.com)
    - **Github Profiles:**
        1. [NSANTRA](https://github.com/NSANTRA)
        2. [ronak-create](https://github.com/ronak-create)

    Feel free to contact us with any questions, feedback, or suggestions about Nexus OCR. Your input is valuable to us and helps improve our learning project!
    """)

if __name__ == "__main__":
    main()