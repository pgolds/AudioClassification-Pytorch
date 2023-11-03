import os
import numpy as np
import torch
import soundfile
import torchaudio

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='./snakers4_silero-vad_master', model='silero_vad',
                              force_reload=False, source='local', onnx=True)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def read_audio1(path: str,
               sampling_rate: int = 16000):

    wav, sr = torchaudio.load(path)
    wav = wav[1].unsqueeze(0)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)

def vad_slice(wav_path):
    wav = read_audio1(wav_path, sampling_rate=8000)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=8000)
    return speech_timestamps, wav

def wav_slice(wav_path):
    fileName = os.path.basename(wav_path).split('_')[0]

    result, rightSound = vad_slice(wav_path)

    index = 0
    for timestamp in result:
        start = timestamp["start"]
        end = timestamp["end"]
        diff = end - start
        if diff > 16000:
            index += 1
            audio = rightSound[start:end]
            soundfile.write(f"wavs/1/{fileName}_{index}.wav", audio, 8000, format="wav")

wav_slice("wavs/1/891023920726532097_13288315396.mp3")
# wav, sr = torchaudio.load("wavs/1/891720255226454017_13242650633.mp3")
# print(wav[0].cpu().numpy())
# w = soundfile.read("wavs/1/891720255226454017_13242650633.mp3")
# print(w)

# print(result)