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
import json

# Load the pre-trained OCR model
@st.cache(allow_output_mutation=True)
def load_ocr_model():
    return load_model("OCR CNN.h5")

# Load the Feedback CSV file
@st.cache(allow_output_mutation=True)
def load_feedback_data():
    feedback_path = os.path.join("feedback_data", "feedback.csv")
    if os.path.exists(feedback_path):
        return pd.read_csv(feedback_path)
    return pd.DataFrame()

def save_feedback_locally(df, img_array, label_list, correct_label):
    """
    Save feedback data locally and create a JSON record for future processing
    """
    try:
        # Prepare the feedback data
        ext_var = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
        ext_var = ext_var.flatten().reshape(1, -1)
        
        # Create DataFrame with new feedback
        columns = [f"pixel{i}" for i in range(1, 785)]
        ext_var_df = pd.DataFrame(ext_var, columns=columns)
        
        # Add the correct label
        ext_var_df["class"] = label_list.index(correct_label if correct_label.isalpha() else int(correct_label))
        
        # Append to main DataFrame
        df_updated = pd.concat([df, ext_var_df], ignore_index=True)
        
        # Save the updated DataFrame
        feedback_dir = "feedback_data"
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Save the main CSV file
        df_updated.to_csv(os.path.join(feedback_dir, "feedback.csv"), index=False)
        
        # Create a timestamped JSON record of this feedback
        feedback_record = {
            "timestamp": datetime.now().isoformat(),
            "correct_label": correct_label,
            "pixel_values": ext_var.tolist(),
            "class_index": int(ext_var_df["class"].iloc[0])
        }
        
        # Save individual feedback record
        record_filename = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(feedback_dir, record_filename), 'w') as f:
            json.dump(feedback_record, f)
            
        return df_updated, True, "Feedback saved successfully"
        
    except Exception as e:
        return df, False, f"Error saving feedback: {str(e)}"

def git_push_changes():
    try:
        # Use HTTPS URL with token
        repo_url = "https://github.com/NSANTRA/Nexus-OCR.git"
        
        # Configure git
        subprocess.run(["git", "config", "--global", "user.email", "neelotpal.santra@gmail.com"])
        subprocess.run(["git", "config", "--global", "user.name", "NSANTRA"])
        
        # Set up the repository if it's not already set up
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"])
            subprocess.run(["git", "remote", "add", "origin", repo_url])
        
        # Fetch and pull latest changes
        subprocess.run(["git", "fetch"])
        subprocess.run(["git", "pull", "origin", "main"])
        
        # Add all files in feedback_data directory
        subprocess.run(["git", "add", "feedback_data/*"])
        
        # Create commit with timestamp
        commit_message = f"Update feedback data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_message])
        
        # Push changes
        subprocess.run(["git", "push", "-u", "origin", "main"])
        
        return True
    except Exception as e:
        st.error(f"Error pushing to git: {str(e)}")
        return False

def handle_feedback_submission(df, img_array, label_list, correct_label):
    """Handle the feedback submission process"""
    df_updated, success, message = save_feedback_locally(df, img_array, label_list, correct_label)
    
    if success:
        # Attempt to push changes to git
        st.info("Attempting to push changes to repository...")
        if git_push_changes():
            st.success(f"""
            Thank you for your feedback! You've entered: {correct_label}
            The feedback has been saved and pushed to the repository successfully.
            You can view all collected feedback in the 'feedback_data' directory.
            """)
        else:
            st.warning(f"""
            Thank you for your feedback! You've entered: {correct_label}
            The feedback has been saved locally but couldn't be pushed to the repository.
            You can view all collected feedback in the 'feedback_data' directory.
            """)
        return df_updated
    else:
        st.error(message)
        return df

def main():
    st.set_page_config(page_title="Nexus OCR", page_icon="📝", layout="wide", initial_sidebar_state="collapsed")
    
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
        img = img / 255.0
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

    def reset_app_state():
        st.session_state.stage = 'draw'
        st.session_state.predicted_label = None
        st.session_state.img_array = None
        st.session_state.canvas_key += 1

    # Canvas options
    st.sidebar.header("Canvas Options")
    stroke_width = st.sidebar.slider("Stroke Width", 1, 50, 15)
    stroke_color = st.sidebar.color_picker("Stroke Color", "#000000")
    bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

    # CSS styling
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

    # Canvas container
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

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            is_correct = st.selectbox(
                "Is the prediction correct?",
                options=["Select an option", "Yes", "No"],
                key="correct_select"
            )
        
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
                        df = handle_feedback_submission(df, st.session_state.img_array, label_list, correct_label)
                    else:
                        st.warning("Please enter a valid single character (0-9 or A-Z).")

            if st.button("Start Over", key="start_over_no"):
                reset_app_state()
                st.experimental_rerun()

    # Contact information
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