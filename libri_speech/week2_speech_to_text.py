import os
import whisper

# Paths
CHUNKS_DIR = "data/audio_chunks"
TRANSCRIPTS_DIR = "data/transcripts"

# Create output folder automatically
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# Load Whisper model (base = good balance)
print("Loading Whisper model")
model = whisper.load_model("base")
print("Whisper model loaded")

# Process each audio chunk
for file in os.listdir(CHUNKS_DIR):
    if file.endswith(".wav"):
        audio_path = os.path.join(CHUNKS_DIR, file)
        transcript_path = os.path.join(TRANSCRIPTS_DIR, file + ".txt")

        print(f" Transcribing: {file}")

        result = model.transcribe(
            audio_path,
            word_timestamps=False
        )

        # Save transcript with timestamps
        with open(transcript_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = round(segment["start"], 2)
                end = round(segment["end"], 2)
                text = segment["text"].strip()
                f.write(f"[{start} - {end}] {text}\n")

print("Week 2 completed: Speech-to-text transcription done")
