import torch
import sounddevice as sd
import time
import numpy as np
from silero import silero_tts


language = "ru" # "de"
model_id = "v3_1_ru" # "v3_de"
sample_rate_speak = 48000

speaker = "eugene" # ['bernd_ungerer', 'eva_k', 'friedrich', 'hokuspokus', 'karlsson', 'random']
put_accent = True
put_yo = True
device = torch.device("cuda")


def speak(origin_text):
    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
    model.to(device)  # gpu or cpu
    text = "hm " + origin_text
    
    
    audio = model.apply_tts(text=text,
                            speaker=speaker,
                            sample_rate=sample_rate_speak,
                            put_accent=put_accent,
                            put_yo=put_yo).numpy()

    del model
    np.save("out.npy", audio)
    conn.send(str(audio.shape[0]).encode())
    sd.play(audio, samplerate=sample_rate_speak)
    time.sleep(len(audio) / sample_rate_speak)
    sd.stop()

import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Listening...")
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024*2).decode()
            if not data:
                break
            speak(data)
            conn.send("done".encode())