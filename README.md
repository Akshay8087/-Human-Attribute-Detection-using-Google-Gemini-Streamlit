# 👤 Human Attribute Detection using Google Gemini & Streamlit

A powerful AI-driven web application that detects **human attributes** from uploaded images using **Google Gemini (Generative AI)** and **Streamlit**.

![App Preview](https://github.com/Akshay8087/your-project-path/assets/preview.gif) <!-- Add screenshot or GIF if available -->

---

## 🚀 Features

- 📸 Upload human images (JPG/PNG)
- 🤖 Detects:
  - Gender, Age, Ethnicity
  - Mood & Facial Expression
  - Hair & Eye Color
  - Glasses, Beard, Headwear
  - Emotions and Confidence Score
- 🧠 Powered by **Gemini 1.5 Flash**
- 📝 Downloadable Analysis Report
- 💻 Built with Streamlit for instant web app UI

---

## 🧠 Demo

Try the live version (if deployed):  
[🔗 Streamlit Cloud / Hugging Face Spaces](#) *(Add your live link here if hosted)*

---

## 📦 Tech Stack

| Tech            | Role                          |
|-----------------|-------------------------------|
| Python 🐍        | Core programming language      |
| Streamlit ⚡     | Web application interface      |
| Google Gemini 🧠 | Image analysis via GenAI       |
| PIL / Pillow 🖼️  | Image processing               |
| dotenv 🔐        | API key management (secure)    |

---

## 🔧 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Akshay8087/human-attribute-detection.git
cd human-attribute-detection
```

2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Set up your Google Gemini API key
You can securely add your API key as an environment variable:
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

Or use a .env file (recommended for local dev):
```bash
GOOGLE_API_KEY=your-gemini-api-key
```
4️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🌟 Show Your Support

If you find this project useful:

- ⭐ **Star** this repo to help others discover it  
- 📣 **Share** it with your friends and network  
- 👀 **Follow** me for more open-source projects and AI tools  

Your support keeps the motivation high! 🙌
