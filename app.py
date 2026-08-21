import os
import re
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel


# ============================================================
# AUDIO TO TEXT
# Generic transcription
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_audio(input_file):

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ).name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        output_file
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Audio conversion failed:\n\n"
            + result.stderr[-4000:]
        )

    return output_file


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_whitespace(text):

    # Only normalize accidental spaces.
    # Do NOT remove repeated words.
    # Do NOT correct words.
    # Do NOT rewrite speech.

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PAGE STYLE
# ============================================================

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

    .info-box {
        padding: 14px;
        border-radius: 10px;
        background: #eff6ff;
        border: 1px solid #93c5fd;
    }

    .warning-box {
        padding: 14px;
        border-radius: 10px;
        background: #fff7ed;
        border: 1px solid #fdba74;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🎙️ Audio to Text</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Transcribe spoken audio into readable text'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

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


    st.markdown(
        """
        **Recommended**

        Sinhala audio:
        `Sinhala`

        Sinhala + English:
        `Auto Detect`

        English:
        `English`
        """
    )


    st.divider()


    st.caption(
        "CPU mode is used for Windows laptops."
    )


# ============================================================
# UPLOAD
# ============================================================

st.header("1. Upload Audio")


uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=SUPPORTED_FORMATS,
    accept_multiple_files=False
)


if uploaded_file is None:

    st.markdown(
        """
        <div class="info-box">

        Upload an audio recording and the application
        will transcribe the speech into text.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# FILE INFORMATION
# ============================================================

extension = Path(
    uploaded_file.name
).suffix.lower()


file_size_mb = (
    uploaded_file.size
    / 1024
    / 1024
)


st.success(
    f"File: {uploaded_file.name}"
)

st.caption(
    f"Size: {file_size_mb:.2f} MB"
)


# ============================================================
# AUDIO PLAYER
# ============================================================

st.header("2. Listen to Audio")


st.audio(
    uploaded_file
)


# ============================================================
# TRANSCRIPTION
# ============================================================

st.header("3. Convert Audio to Text")


if st.button(
    "🎙️ TRANSCRIBE",
    type="primary",
    use_container_width=True
):

    original_file = None
    wav_file = None


    try:

        # ----------------------------------------------------
        # Save uploaded audio
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temporary:

            temporary.write(
                uploaded_file.getbuffer()
            )

            original_file = temporary.name


        # ----------------------------------------------------
        # Convert audio
        # ----------------------------------------------------

        with st.status(
            "Preparing audio...",
            expanded=False
        ):

            wav_file = convert_audio(
                original_file
            )


        # ----------------------------------------------------
        # Load Whisper
        # ----------------------------------------------------

        with st.status(
            f"Loading {model_name} model...",
            expanded=False
        ):

            model = load_model(
                model_name
            )


        # ----------------------------------------------------
        # Selected language
        # ----------------------------------------------------

        language_code = LANGUAGES[
            language_name
        ]


        # ----------------------------------------------------
        # TRANSCRIPTION
        # ----------------------------------------------------

        with st.status(
            "Transcribing audio...",
            expanded=True
        ):

            segments, info = model.transcribe(

                wav_file,

                language=language_code,

                task="transcribe",

                # No medical prompt.
                # No forced vocabulary.
                initial_prompt=None,

                beam_size=5,

                best_of=1,

                temperature=0,

                # Prevent previous hallucinated text
                # from influencing the next segment.
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


            transcript_lines = []

            uncertain_segments = []


            for segment in segments:

                text = normalize_whitespace(
                    segment.text
                )


                if not text:
                    continue


                # ------------------------------------------------
                # IMPORTANT:
                # DO NOT remove repetitions.
                # If speaker says:
                # "yes yes yes"
                # it stays:
                # "yes yes yes"
                # ------------------------------------------------


                confidence = segment.avg_logprob


                if confidence < -1.3:

                    uncertain_segments.append(
                        text
                    )

                    transcript_lines.append(
                        "[uncertain] " + text
                    )

                else:

                    transcript_lines.append(
                        text
                    )


        transcript = "\n".join(
            transcript_lines
        ).strip()


        if not transcript:

            transcript = (
                "[No intelligible speech detected.]"
            )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        st.session_state[
            "transcript"
        ] = transcript


        st.session_state[
            "filename"
        ] = uploaded_file.name


        st.session_state[
            "detected_language"
        ] = info.language


        st.session_state[
            "language_probability"
        ] = info.language_probability


        st.session_state[
            "uncertain_count"
        ] = len(
            uncertain_segments
        )


        st.success(
            "Transcription completed."
        )


    except FileNotFoundError:

        st.error(
            "FFmpeg is not installed or cannot be found."
        )

        st.info(
            "Install FFmpeg, restart PowerShell, "
            "and run the application again."
        )


    except Exception as error:

        st.error(
            "Transcription failed."
        )

        st.exception(
            error
        )


    finally:

        # ----------------------------------------------------
        # Delete temporary files
        # ----------------------------------------------------

        for filename in [
            original_file,
            wav_file
        ]:

            if filename:

                try:

                    os.remove(
                        filename
                    )

                except Exception:

                    pass


# ============================================================
# TRANSCRIPT
# ============================================================

if "transcript" in st.session_state:

    st.header("4. Transcript")


    detected_language = (
        st.session_state.get(
            "detected_language",
            "unknown"
        )
    )


    language_probability = (
        st.session_state.get(
            "language_probability",
            0
        )
    )


    st.info(
        "Detected language: "
        + str(detected_language)
        + " | Confidence: "
        + f"{language_probability:.0%}"
    )


    uncertain_count = (
        st.session_state.get(
            "uncertain_count",
            0
        )
    )


    if uncertain_count > 0:

        st.warning(
            f"{uncertain_count} segment(s) were "
            "marked [uncertain]. Listen to those "
            "sections in the original audio."
        )


    transcript = st.text_area(
        "Transcribed text",
        value=st.session_state[
            "transcript"
        ],
        height=550
    )


    st.session_state[
        "transcript"
    ] = transcript


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    word_count = len(
        transcript.split()
    )


    character_count = len(
        transcript
    )


    st.caption(
        f"Words: {word_count} | "
        f"Characters: {character_count}"
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    st.header("5. Download")


    original_name = Path(
        st.session_state[
            "filename"
        ]
    ).stem


    output_name = (
        original_name
        + "_transcript.txt"
    )


    st.download_button(

        label="⬇️ Download TXT",

        data=transcript.encode(
            "utf-8"
        ),

        file_name=output_name,

        mime="text/plain;charset=utf-8",

        use_container_width=True
    )


# ============================================================
# INFORMATION
# ============================================================

st.divider()


st.markdown(
    """
    <div class="warning-box">

    <b>Transcription behavior</b><br><br>

    This application attempts to reproduce the speech in the
    recording. It does not intentionally translate, summarize,
    correct grammar, or interpret the content.

    Words that are genuinely unclear may be marked
    <b>[uncertain]</b> rather than being silently presented
    as certain.

    </div>
    """,
    unsafe_allow_html=True
)
