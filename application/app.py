import streamlit as st
import google.generativeai as genai
import os
import PIL.Image
import datetime

# 🌐 Set API Key securely (better to use secrets in production)
os.environ["GOOGLE_API_KEY"] = "Add Your API Key Here OK"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🎯 Load the Gemini Model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# 📦 Function to analyze human attributes
def analyze_human_attributes(image):
    prompt = """
    You are an AI trained to analyze human attributes from images with high accuracy. 
    Carefully analyze the given image and return the following structured details:

    You have to return all results as you have the image, don't want any apologize or empty results.

    - **Gender** (Male/Female/Non-binary)
    - **Age Estimate** (e.g., 25 years)
    - **Ethnicity** (e.g., Asian, Caucasian, African, etc.)
    - **Mood** (e.g., Happy, Sad, Neutral, Excited)
    - **Facial Expression** (e.g., Smiling, Frowning, Neutral, etc.)
    - **Glasses** (Yes/No)
    - **Beard** (Yes/No)
    - **Hair Color** (e.g., Black, Blonde, Brown)
    - **Eye Color** (e.g., Blue, Green, Brown)
    - **Headwear** (Yes/No, specify type if applicable)
    - **Emotions Detected** (e.g., Joyful, Focused, Angry, etc.)
    - **Confidence Level** (Accuracy of prediction in percentage)
    """
    try:
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"

# 🧭 Sidebar
with st.sidebar:
    st.title("👤 Human Analyzer AI")
    st.info("Upload an image and get detailed human attribute predictions using Google Gemini.")
    st.markdown("---")
    st.markdown("📍 Created by **Akshay Rathod**")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/akshaygr1/)")
    st.markdown("💻 [GitHub](https://github.com/Akshay8087)")

# 🖼️ Main UI
st.title("🧠 Human Attribute Detection App")
st.write("Upload an image to detect human features using Google's Gemini AI.")

uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing image using Gemini AI..."):
            results = analyze_human_attributes(img)
        
        st.success("✅ Analysis Complete!")
        st.markdown(results)

        # Option to download the result as text
        report_name = f"Human_Analysis_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button("📥 Download Report", results, file_name=report_name)

        # Optional: expandable for raw Gemini output
        with st.expander("🔍 Show Raw Analysis"):
            st.code(results)
