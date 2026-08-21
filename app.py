import os
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


st.set_page_config(
    page_title="Audio to Text",
    page_icon="🎙️",
    layout="wide"
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


LANGUAGES = {
    "Auto Detect": None,
    "English": "en",
    "Sinhala": "si",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Urdu": "ur",
}


@st.cache_resource
def load_model(model_name):

    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(
            1,
            (os.cpu_count() or 4) - 1
        ),
        num_workers=1
    )


def transcribe_audio(
    uploaded_file,
    model_name,
    language_code
):

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    try:

        temporary_file.write(
            uploaded_file.getbuffer()
        )

        temporary_file.close()

        model = load_model(
            model_name
        )

        segments, info = model.transcribe(

            temporary_file.name,

            language=language_code,

            task="transcribe",

            initial_prompt=None,

            beam_size=5,

            best_of=1,

            temperature=0,

            condition_on_previous_text=False,

            compression_ratio_threshold=2.4,

            log_prob_threshold=-1.0,

            no_speech_threshold=0.60,

            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 250
            },

            word_timestamps=False
        )

        lines = []

        for segment in segments:

            text = segment.text.strip()

            if text:

                lines.append(text)

        transcript = "\n".join(
            lines
        ).strip()

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

        try:

            os.remove(
                temporary_file.name
            )

        except Exception:

            pass


st.markdown(
    """
    <style>

    .title {
        font-size: 42px;
        font-weight: 700;
        color: #17365d;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="title">🎙️ Audio to Text</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Whatever is spoken in the audio → text'
    '</div>',
    unsafe_allow_html=True
)


with st.sidebar:

    st.header("Transcription Settings")

    model_name = st.selectbox(
        "Whisper Model",
        [
            "medium",
            "small"
        ],
        index=0
    )

    language_name = st.selectbox(
        "Audio Language",
        list(LANGUAGES.keys()),
        index=0
    )

    st.divider()

    st.write(
        "The application does not translate, "
        "summarize, or add medical terminology."
    )


st.header("1. Upload Audio")


uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=SUPPORTED_FORMATS
)


if uploaded_file is None:

    st.info(
        "Upload an audio recording to begin."
    )

    st.stop()


st.success(
    f"Uploaded: {uploaded_file.name}"
)


st.caption(
    f"Size: {uploaded_file.size / 1024 / 1024:.2f} MB"
)


st.header("2. Listen")


st.audio(
    uploaded_file
)


st.header("3. Transcribe")


if st.button(
    "🎙️ TRANSCRIBE",
    type="primary",
    use_container_width=True
):

    language_code = LANGUAGES[
        language_name
    ]

    try:

        with st.spinner(
            "Loading transcription model..."
        ):

            with st.spinner(
                "Transcribing audio..."
            ):

                (
                    transcript,
                    detected_language,
                    language_probability
                ) = transcribe_audio(
                    uploaded_file,
                    model_name,
                    language_code
                )


        st.session_state[
            "transcript"
        ] = transcript

        st.session_state[
            "detected_language"
        ] = detected_language

        st.session_state[
            "language_probability"
        ] = language_probability

        st.session_state[
            "filename"
        ] = uploaded_file.name

        st.success(
            "Transcription completed."
        )

    except Exception as error:

        st.error(
            "Transcription failed."
        )

        st.exception(error)


if "transcript" in st.session_state:

    st.header("4. Transcript")

    detected = st.session_state[
        "detected_language"
    ]

    probability = st.session_state[
        "language_probability"
    ]

    st.info(
        f"Detected language: {detected} | "
        f"Confidence: {probability:.0%}"
    )

    transcript = st.text_area(
        "Transcribed text",
        value=st.session_state["transcript"],
        height=550
    )

    st.session_state[
        "transcript"
    ] = transcript

    word_count = len(
        transcript.split()
    )

    st.caption(
        f"Words: {word_count}"
    )

    original_name = Path(
        st.session_state["filename"]
    ).stem

    output_name = (
        original_name +
        "_transcript.txt"
    )

    st.download_button(
        "⬇️ Download TXT",
        data=transcript.encode("utf-8"),
        file_name=output_name,
        mime="text/plain;charset=utf-8",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "The transcript attempts to preserve what was "
        "spoken. It does not intentionally translate, "
        "summarize, or medically interpret the recording."
    )