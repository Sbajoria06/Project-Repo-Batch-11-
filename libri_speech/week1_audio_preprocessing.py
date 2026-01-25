import os
import librosa
import soundfile as sf
from pydub import AudioSegment, effects, silence

RAW_AUDIO_DIR = "data/raw_audio/dev-clean"
PROCESSED_DIR = "data/processed_audio"
CHUNKS_DIR = "data/audio_chunks"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

for root, dirs, files in os.walk(RAW_AUDIO_DIR):
    for file in files:
        if file.endswith(".flac"):
            flac_path = os.path.join(root, file)

            # Load & resample
            y, sr = librosa.load(flac_path, sr=16000, mono=True)
            wav_name = file.replace(".flac", ".wav")
            processed_path = os.path.join(PROCESSED_DIR, wav_name)
            sf.write(processed_path, y, 16000)

            audio = AudioSegment.from_wav(processed_path)
            audio = effects.normalize(audio)

            chunks = silence.split_on_silence(
                audio,
                min_silence_len=1000,
                silence_thresh=-40
            )

            for i, chunk in enumerate(chunks):
                if len(chunk) >= 20000:
                    chunk.export(
                        f"{CHUNKS_DIR}/{wav_name[:-4]}_chunk{i}.wav",
                        format="wav"
                    )

print("Week 1: LibriSpeech preprocessing completed")
