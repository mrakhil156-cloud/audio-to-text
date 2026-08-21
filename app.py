import os
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


st.set_page_config(
    page_title="Audio to Text",
    page_icon="🎙️",
    layout="centered"
)


SUPPORTED_FORMATS = [
    "ogg",
    "oga",
    "wav",
    "mp3",
    "m4a",
    "aac",
    "flac",
    "webm",
    "wma",
    "mp4",
    "mpeg",
    "mpga"
]


# ---------------------------------------------------------
# LOAD ONE MODEL ONLY
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():

    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1
    )


# ---------------------------------------------------------
# TRANSCRIPTION
# ---------------------------------------------------------

def transcribe_audio(uploaded_file, language=None):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp.name


        model = load_model()


        segments, info = model.transcribe(
            temp_path,

            language=language,

            task="transcribe",

            beam_size=1,

            best_of=1,

            temperature=0,

            condition_on_previous_text=False,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200
            },

            word_timestamps=False
        )


        lines = []

        for segment in segments:

            text = segment.text.strip()

            if text:

                lines.append(text)


        transcript = "\n".join(lines).strip()


        if not transcript:

            transcript = (
                "[No intelligible speech detected.]"
            )


        return (
            transcript,
            info.language,
            info.language_probability
        )


    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------

st.title("🎙️ Audio to Text")

st.write(
    "Upload an audio recording and convert "
    "the spoken content into text."
)


st.info(
    "The application transcribes what is spoken. "
    "It does not intentionally summarize or translate the audio."
)


# ---------------------------------------------------------
# LANGUAGE
# ---------------------------------------------------------

language_options = {
    "Auto Detect": None,
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Sinhala": "si",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur"
}


language_name = st.selectbox(
    "Audio language",
    list(language_options.keys())
)


language_code = language_options[
    language_name
]


# ---------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload audio",
    type=SUPPORTED_FORMATS
)


if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )


    st.audio(
        uploaded_file
    )


    if st.button(
        "🎙️ Transcribe",
        type="primary",
        use_container_width=True
    ):

        # Prevent accidental double-clicks
        if st.session_state.get(
            "transcribing",
            False
        ):

            st.warning(
                "Transcription is already running."
            )

            st.stop()


        st.session_state[
            "transcribing"
        ] = True


        try:

            progress = st.progress(0)

            status = st.empty()


            status.info(
                "Loading transcription engine..."
            )

            progress.progress(10)


            status.info(
                "Transcribing audio..."
            )

            transcript, detected_language, confidence = (
                transcribe_audio(
                    uploaded_file,
                    language_code
                )
            )


            progress.progress(100)


            status.success(
                "Transcription completed."
            )


            st.session_state[
                "transcript"
            ] = transcript

            st.session_state[
                "detected_language"
            ] = detected_language

            st.session_state[
                "confidence"
            ] = confidence

            st.session_state[
                "filename"
            ] = uploaded_file.name


        except Exception as e:

            st.error(
                "Transcription failed."
            )

            st.exception(e)


        finally:

            st.session_state[
                "transcribing"
            ] = False


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

if "transcript" in st.session_state:

    st.divider()

    st.subheader(
        "Transcript"
    )


    detected_language = st.session_state[
        "detected_language"
    ]

    confidence = st.session_state[
        "confidence"
    ]


    st.caption(
        f"Detected language: "
        f"{detected_language} | "
        f"Detection confidence: "
        f"{confidence:.0%}"
    )


    transcript = st.text_area(
        "Transcribed text",
        value=st.session_state["transcript"],
        height=500
    )


    st.session_state[
        "transcript"
    ] = transcript


    word_count = len(
        transcript.split()
    )


    st.caption(
        f"Word count: {word_count}"
    )


    original_name = Path(
        st.session_state["filename"]
    ).stem


    output_filename = (
        original_name +
        "_transcript.txt"
    )


    st.download_button(
        "⬇️ Download TXT",
        data=transcript.encode("utf-8"),
        file_name=output_filename,
        mime="text/plain",
        use_container_width=True
    )