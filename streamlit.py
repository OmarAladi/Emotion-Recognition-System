import streamlit as st
import cv2
import os
import tempfile
from facial_expression_analysis import process_emotion_frame
from datetime import datetime
import time

# Streamlit WebRTC for live streaming
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Page configuration
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎥",
    layout="centered"
)

# Dark theme CSS
st.markdown(
    """
    <style>
      .stApp { background-color: #000; color: #fff; }
      .stSidebar { background-color: #111; }
      h1 { text-align: center; margin-top: 20px; }
      .stButton>button { background-color: #333; color: #fff; border-radius: 8px; padding: 8px 16px; }
      .stButton>button:hover { background-color: #555; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1>Emotion Recognition System</h1>", unsafe_allow_html=True)

# Mode selection
mode = st.sidebar.radio("Mode", ["Live Camera", "Process Video"])

if mode == "Live Camera":
    # Transformer for live camera frames
    class EmotionTransformer(VideoTransformerBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = process_emotion_frame(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="emotion-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )

else:
    # Video processing mode
    uploaded = st.sidebar.file_uploader(
        "Upload Video (mp4, avi, mov)", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp_file.write(uploaded.read())
        tmp_file.close()

        cap = cv2.VideoCapture(tmp_file.name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"output_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed = 0
        placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_emotion_frame(frame)
            out.write(frame)
            processed += 1
            progress.progress(min(processed / total_frames, 1.0))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(img_rgb, use_container_width=True)
            time.sleep(0.01)

        cap.release()
        out.release()

        st.success("✅ Processing complete!")
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name=os.path.basename(out_path),
                mime="video/mp4"
            )
        st.video(out_path)
        os.unlink(tmp_file.name)
