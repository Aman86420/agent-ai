import streamlit as st
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
        genai.configure(api_key=api_key)
        st.success("API Key configured!")

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
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )

        if feedback:
            prompt_template = """You are an expert content script writer. 

Previous script feedback: {feedback}

Topic: {topic}
Goal: {goal}

Based on the feedback above, revise and improve the script. Create an engaging, well-structured video script that addresses the feedback.

Format the script with:
- Hook/Introduction
- Main Content (with clear sections)
- Call to Action/Conclusion

Script:"""
            prompt = PromptTemplate(
                input_variables=["topic", "goal", "feedback"],
                template=prompt_template
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(topic=topic, goal=goal, feedback=feedback)
        else:
            prompt_template = """You are an expert content script writer. Create an engaging video script.

Topic: {topic}
Goal: {goal}

Create a compelling, well-structured video script (approximately 60-90 seconds when spoken).

Format the script with:
- Hook/Introduction (5-10 seconds)
- Main Content (40-60 seconds with clear sections)
- Call to Action/Conclusion (10-15 seconds)

Make it engaging, conversational, and optimized for viewer retention.

Script:"""
            prompt = PromptTemplate(
                input_variables=["topic", "goal"],
                template=prompt_template
            )
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(topic=topic, goal=goal)

        return result
    except Exception as e:
        st.error(f"Error generating script: {str(e)}")
        return None


def generate_thumbnail(topic, goal, feedback=None):
    """Generate thumbnail using Gemini Image API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        if feedback:
            prompt = f"""Create a compelling YouTube thumbnail image for:
Topic: {topic}
Goal: {goal}

Previous feedback: {feedback}

Generate an eye-catching, professional thumbnail that addresses the feedback. 
Include bold text overlay if relevant, vibrant colors, and high contrast for visibility."""
        else:
            prompt = f"""Create a compelling YouTube thumbnail image for:
Topic: {topic}
Goal: {goal}

Generate an eye-catching, professional thumbnail with:
- Bold, readable text overlay (if relevant)
- Vibrant, high-contrast colors
- Clear focal point
- Professional design that stands out in search results"""

        # Note: Gemini's image generation is limited. Using text-based description
        # In production, you'd integrate with an image generation API
        response = model.generate_content(prompt)

        # For demo purposes, return description
        # In production, integrate with actual image generation API
        return response.text

    except Exception as e:
        st.error(f"Error generating thumbnail: {str(e)}")
        return None


def generate_video_metadata(script, thumbnail_desc):
    """Generate video metadata and simulation"""
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')

        prompt = f"""Based on this script and thumbnail description, create video metadata:

Script: {script[:500]}...

Thumbnail: {thumbnail_desc[:200]}...

Provide:
1. Video Title (SEO optimized, under 60 characters)
2. Video Description (detailed, 200-300 words)
3. Tags (10-15 relevant tags)
4. Best posting time recommendation
5. Target audience description

Format as structured sections."""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        st.error(f"Error generating video metadata: {str(e)}")
        return None


# Main app
st.title("🎬 AI Content Generation Pipeline")
st.markdown("Generate complete video content: Script → Thumbnail → Video")

# Stage 1: User Input
if st.session_state.stage == 'input':
    st.header("1️⃣ Content Planning")

    col1, col2 = st.columns(2)

    with col1:
        topic = st.text_input(
            "🎯 Video Topic",
            value=st.session_state.topic,
            placeholder="e.g., 'How to use AI for productivity'",
            help="What is your video about?"
        )

    with col2:
        goal = st.text_input(
            "🎪 Content Goal",
            value=st.session_state.goal,
            placeholder="e.g., 'Educate beginners on AI tools'",
            help="What do you want to achieve?"
        )

    if st.button("Generate Script →", type="primary", disabled=not (topic and goal and api_key)):
        st.session_state.topic = topic
        st.session_state.goal = goal
        st.session_state.stage = 'script'
        st.rerun()

    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to continue")

# Stage 2: Script Generation
elif st.session_state.stage == 'script':
    st.header("2️⃣ Script Generation")

    with st.spinner("🤖 Generating script with AI..."):
        script = generate_script(
            st.session_state.topic,
            st.session_state.goal,
            st.session_state.script_feedback if st.session_state.script_feedback else None
        )

        if script:
            st.session_state.script = script
            st.session_state.stage = 'script_review'
            st.rerun()

# Stage 3: Script Review
elif st.session_state.stage == 'script_review':
    st.header("3️⃣ Script Review")

    st.subheader("📝 Generated Script")
    st.markdown(f"**Topic:** {st.session_state.topic}")
    st.markdown(f"**Goal:** {st.session_state.goal}")

    st.markdown("---")
    st.markdown(st.session_state.script)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Finalize Script?")
        decision = st.radio(
            "Choose an option:",
            ["✅ Yes, finalize script", "✏️ No, I want to edit"],
            key="script_decision"
        )

    if decision == "✏️ No, I want to edit":
        st.session_state.script_feedback = st.text_area(
            "What changes would you like?",
            placeholder="e.g., 'Make it more casual', 'Add more examples', 'Shorten the intro'...",
            height=100
        )

        if st.button("🔄 Regenerate Script", type="primary"):
            if st.session_state.script_feedback:
                st.session_state.stage = 'script'
                st.rerun()
            else:
                st.warning("Please provide feedback for regeneration")
    else:
        if st.button("Continue to Thumbnail →", type="primary"):
            st.session_state.stage = 'thumbnail'
            st.session_state.script_feedback = ""
            st.rerun()

# Stage 4: Thumbnail Generation
elif st.session_state.stage == 'thumbnail':
    st.header("4️⃣ Thumbnail Generation")

    with st.spinner("🎨 Generating thumbnail design..."):
        thumbnail = generate_thumbnail(
            st.session_state.topic,
            st.session_state.goal,
            st.session_state.thumbnail_feedback if st.session_state.thumbnail_feedback else None
        )

        if thumbnail:
            st.session_state.thumbnail = thumbnail
            st.session_state.stage = 'thumbnail_review'
            st.rerun()

# Stage 5: Thumbnail Review
elif st.session_state.stage == 'thumbnail_review':
    st.header("5️⃣ Thumbnail Review")

    st.subheader("🎨 Thumbnail Design Concept")
    st.info("""**Note:** This is a design description. In production, this would generate an actual image 
using services like DALL-E, Midjourney API, or Stable Diffusion.""")

    st.markdown(st.session_state.thumbnail)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Finalize Thumbnail?")
        decision = st.radio(
            "Choose an option:",
            ["✅ Yes, finalize thumbnail", "✏️ No, I want changes"],
            key="thumbnail_decision"
        )

    if decision == "✏️ No, I want changes":
        st.session_state.thumbnail_feedback = st.text_area(
            "What changes would you like?",
            placeholder="e.g., 'Use brighter colors', 'Add more text', 'Different style'...",
            height=100
        )

        if st.button("🔄 Regenerate Thumbnail", type="primary"):
            if st.session_state.thumbnail_feedback:
                st.session_state.stage = 'thumbnail'
                st.rerun()
            else:
                st.warning("Please provide feedback for regeneration")
    else:
        if st.button("Generate Final Video →", type="primary"):
            st.session_state.stage = 'video'
            st.session_state.thumbnail_feedback = ""
            st.rerun()

# Stage 6: Video Generation
elif st.session_state.stage == 'video':
    st.header("6️⃣ Video Generation & Metadata")

    with st.spinner("🎬 Generating video metadata and optimization..."):
        video_metadata = generate_video_metadata(
            st.session_state.script,
            st.session_state.thumbnail
        )

        if video_metadata:
            st.session_state.video_metadata = video_metadata
            st.session_state.stage = 'complete'
            st.rerun()

# Stage 7: Complete
elif st.session_state.stage == 'complete':
    st.header("✅ Content Generation Complete!")
    st.balloons()

    tab1, tab2, tab3, tab4 = st.tabs(["📝 Script", "🎨 Thumbnail", "🎬 Video Info", "📊 Summary"])

    with tab1:
        st.subheader("Final Script")
        st.markdown(st.session_state.script)
        st.download_button(
            "📥 Download Script",
            st.session_state.script,
            file_name="video_script.txt",
            mime="text/plain"
        )

    with tab2:
        st.subheader("Thumbnail Design")
        st.markdown(st.session_state.thumbnail)
        st.info("In production, you would have an actual image file to download here")

    with tab3:
        st.subheader("Video Metadata & Optimization")
        st.markdown(st.session_state.video_metadata)

        st.info("""**Note:** Actual video generation would require:
- Video editing API (e.g., Synthesia, D-ID, Pictory)
- Voice synthesis (e.g., ElevenLabs, Google TTS)
- Video assembly pipeline

This demo shows the complete workflow structure.""")

    with tab4:
        st.subheader("Project Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Topic", st.session_state.topic)
            st.metric("Goal", st.session_state.goal)

        with col2:
            st.metric("Script Length", f"{len(st.session_state.script.split())} words")
            st.metric("Status", "✅ Complete")

    if st.button("🔄 Create New Content", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with LangChain, Streamlit & Google Gemini | 
    <a href='https://github.com' target='_blank'>GitHub</a> | 
    <a href='https://docs.google.com/gemini' target='_blank'>Gemini API Docs</a></p>
</div>
""", unsafe_allow_html=True)