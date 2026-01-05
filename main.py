import streamlit as st
import google.generativeai as genai
from langchain import PromptTemplate, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="AI Content Generation Pipeline",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 'input'
if 'script' not in st.session_state:
    st.session_state.script = None
if 'thumbnail' not in st.session_state:
    st.session_state.thumbnail = None
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'goal' not in st.session_state:
    st.session_state.goal = ""
if 'script_feedback' not in st.session_state:
    st.session_state.script_feedback = ""
if 'thumbnail_feedback' not in st.session_state:
    st.session_state.thumbnail_feedback = ""

# Sidebar for API key
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password", key="api_key")

    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.success("API Key configured!")
        except Exception:
            # older/newer packages may differ; we'll still store the key
            st.warning("API key provided (could not configure programmatically).")

    st.markdown("---")
    st.markdown("### 📊 Pipeline Progress")

    stages = {
        'input': '1️⃣ Topic Input',
        'script': '2️⃣ Script Generation',
        'script_review': '3️⃣ Script Review',
        'thumbnail': '4️⃣ Thumbnail Generation',
        'thumbnail_review': '5️⃣ Thumbnail Review',
        'video': '6️⃣ Video Generation',
        'complete': '✅ Complete'
    }

    for stage_key, stage_name in stages.items():
        if st.session_state.stage == stage_key:
            st.markdown(f"**{stage_name}** ← Current")
        else:
            st.markdown(stage_name)

    st.markdown("---")
    if st.button("🔄 Reset Pipeline", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# Helper functions
def generate_script(topic, goal, feedback=None):
    """Generate video script using Gemini via LangChain"""
    try:
        # Use API key from sidebar input if available
        google_api_key = st.session_state.get('api_key') or os.environ.get('GENAI_API_KEY')

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )

        if feedback:
            prompt_template = (
                "You are an expert content script writer.\n\n"
                "Previous script feedback: {feedback}\n\n"
                "Topic: {topic}\n"
                "Goal: {goal}\n\n"
                "Based on the feedback above, revise and improve the script. Create an engaging, well-structured video script that addresses the feedback.\n\n"
                "Format the script with:\n"
                "- Hook/Introduction\n"
                "- Main Content (with clear sections)\n"
                "- Call to Action/Conclusion\n\n"
                "Script:"
            )
            prompt = PromptTemplate(input_variables=["topic", "goal", "feedback"], template=prompt_template)
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(topic=topic, goal=goal, feedback=feedback)
        else:
            prompt_template = (
                "You are an expert content script writer. Create an engaging video script.\n\n"
                "Topic: {topic}\n"
                "Goal: {goal}\n\n"
                "Format the script with:\n"
                "- Hook/Introduction\n"
                "- Main Content (with clear sections)\n"
                "- Call to Action/Conclusion\n\n"
                "Script:"
            )
            prompt = PromptTemplate(input_variables=["topic", "goal"], template=prompt_template)
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(topic=topic, goal=goal)

        return result
    except Exception as e:
        st.error(f"Script generation failed: {e}")
        return None


# Minimal UI logic to use the helper
st.title("AI Content Generation Pipeline")

if st.session_state.stage == 'input':
    st.header("1. Topic Input")
    topic = st.text_input("Topic", value=st.session_state.topic)
    goal = st.text_input("Goal / Audience / Style", value=st.session_state.goal)

    if st.button("Generate Script"):
        st.session_state.topic = topic
        st.session_state.goal = goal
        with st.spinner("Generating script..."):
            script = generate_script(topic, goal)
        if script:
            st.session_state.script = script
            st.session_state.stage = 'script'
            st.experimental_rerun()

elif st.session_state.stage == 'script':
    st.header("2. Script Generation")
    st.subheader("Generated Script")
    st.text_area("Script", value=st.session_state.script or "", height=400)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Regenerate Script"):
            with st.spinner("Regenerating..."):
                script = generate_script(st.session_state.topic, st.session_state.goal)
            if script:
                st.session_state.script = script
                st.experimental_rerun()
    with col2:
        if st.button("Proceed to Thumbnail"):
            st.session_state.stage = 'thumbnail'
            st.experimental_rerun()

else:
    st.info("Pipeline stage not implemented in this minimal version.")
