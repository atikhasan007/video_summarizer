import os

# -------------------------------
# Set API key in environment
# Must be before importing Phi
# -------------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyCysXxnTkx0Ccg58_WULbhyn8IpMQNHYsk"

import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import time
from pathlib import Path
import tempfile

# -------------------------------
# Streamlit page configuration
# -------------------------------
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🤖",
    layout="wide"
)

st.title("Phidata Video AI Summarizer Agent 🤖")
st.header("Powered by Gemini 2.0 Flash Exp")

# -------------------------------
# Create Gemini model with API key
# -------------------------------
gemini_model = Gemini(id="gemini-2.5-flash", api_key=os.environ["GOOGLE_API_KEY"])

# -------------------------------
# Initialize Phi Agent
# -------------------------------
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=gemini_model,
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodel_Agent = initialize_agent()

# -------------------------------
# File uploader
# -------------------------------
video_file = st.file_uploader(
    "Upload a video file", 
    type=['mp4','mov','avi'],
    help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional information.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}
                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    response = multimodel_Agent.run(analysis_prompt, videos=[processed_video])

                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)

else:
    st.info("Upload a video file to begin analysis.")

# -------------------------------
# Customize text area height
# -------------------------------
st.markdown(
    """
<style>
.stTextArea textarea{
    height: 100px;
}
</style>
""",
    unsafe_allow_html=True
)
