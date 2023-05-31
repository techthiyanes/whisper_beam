from scipy.io import wavfile
import numpy as np
# import librosa
import whisper
import ftfy
import os
import torch
# import json 
import secrets
# import subprocess

# def process_audio(**inputs):
#     audio = inputs["audio"]
#     audio["data"] = np.array(audio["data"], dtype=np.int16)
#     token = secrets.token_hex(4)
#     wavfile.write(f"./{token}.wav", **audio)
#     subprocess.run(f"whisper.cpp/main -f ./{token}.wav -of testing --language it")
#     return {"response": "OK"}

# from transformers import AutoProcessor, WhisperForConditionalGeneration


def load_processor_model(lang="it"):
    # processor = AutoProcessor.from_pretrained("openai/whisper-tiny", language="italian")
    # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    # return processor, model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device", device)
    
    model = whisper.load_model(
        os.environ.get("WHISPER_MODEL", "tiny"), 
        download_root="/workspace/cache_data/.cache"
    )

    return model


def process_audio(**inputs):
    audio = inputs["audio"]
    audio["data"] = np.array(audio["data"], dtype=np.int16)
    token = secrets.token_hex(4)
    wavfile.write(f"./{token}.wav", **audio)

    model = load_processor_model()
    
    # Simple processing
    # transcription = model.transcribe(f"./{token}.wav", language="it")

    # Using Huggingface
    # audio["data"] = audio["data"][:, 0].astype(np.float32)
    # original_sr = audio["rate"]
    # audio_16k = librosa.resample(audio["data"], orig_sr=original_sr, target_sr=16000)
    # processor, model = load_processor_model()
    # inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    # input_features = inputs.input_features
    # generated_ids = model.generate(inputs=input_features)
    # transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(f"./{token}.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # text = ftfy.fix_text(transcription["text"])
    text = ftfy.fix_text(result.text)
    print(text)
    return {"response": text}


if __name__ == "__main__":
    pass
    # text = "Testing 123"
    # samplerate, data = wavfile.read('record.wav')
    # d = {
    #     "rate": samplerate,
    #     "data": data
    # }
    # process_audio(data=json.dumps(d))
