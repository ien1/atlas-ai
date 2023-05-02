import math
import pygame as pg
import numpy as np
from scipy.signal import resample
import sounddevice as sd
import threading
import webrtcvad
import whisper
from scipy.io import wavfile
import openai
import socket
import cv2


## utils
import datetime
from num2words import num2words
import os
import webbrowser

class Visualizer:
    def __init__(self) -> None:

        ### Display settings ###
        self.active = False
        self.done = False

        ### Constants ###
        self.BLACK = (0, 0, 0)
        self.SIZE = [400, 400]


        ### Visualization objects ###
        self.circle = self.create_circle()
        self.POINTS = self.circle.shape[0]

        self.data = None

        ### TTS objects ###
        self.spoken_bool = False
        self.spoken = None

        self.spoken_circle = self.create_circle(r=25)
        self.spoken_POINTS = self.circle.shape[0]

        self.spoken_frame = np.ones(shape=(self.spoken_POINTS))

        self.last = 0

        ### Loading animation objects ###
        self.loaded = False
        self.images = []
        for num in range(1, 61):
            img = pg.image.load(f"./assets/img/loading/loading_40ms_{num}.png")
            self.images.append(img)
        
        self.index = 0
        self.counter = 0
        self.speed = 3

        ### OPENCV test img ###
        self.cv2_img = np.zeros((400, 400, 3), np.uint8)

    def create_circle(self, a=0, b=0, r=100):
        #The lower this value the higher quality the circle is with more points generated
        stepSize = 0.01
        #Generated vertices
        positions = []
        t = 0
        while t < 2 * math.pi:
            positions.append((r * math.cos(t) + a, r * math.sin(t) + b))
            t += stepSize
        return np.array(positions)

    def start(self):
        pg.init()

        ### PyGame objects ###
        self.screen = pg.display.set_mode(self.SIZE)
        self.clock = pg.time.Clock()
        while not self.done:
            for event in pg.event.get(): 
                if event.type == pg.QUIT: 
                    self.done = True
            self.screen.fill(self.BLACK)
            if self.loaded:
                self.draw()
            else:
                self.load()
            pg.display.flip()
            self.clock.tick(30)
        pg.quit()
    
    def load(self):
        if self.counter == self.speed:
            self.index += 1
            self.counter = 0
        if self.index >= len(self.images):
            self.index = 0
        img = self.images[self.index]
        rect = img.get_rect()
        rect.center = (200, 200)
        self.screen.blit(img, rect)
        self.counter += 1
    

    def draw(self):
        # bright blue: 117,251,251
        # darker blue: 31,101,104

        # bright red: 251,117,117
        # darker red: 104,34,31
        if self.spoken_bool:
            num_elements = int((1 / self.clock.get_fps()) * 48000)
            if num_elements + self.last <= self.spoken.shape[0]:
                self.spoken_frame = self.spoken[self.last:self.last+num_elements]
                self.spoken_frame = resample(self.spoken_frame, self.spoken_POINTS) + 1
                self.last += num_elements
            else:
                self.spoken_bool = False
                self.last = 0
        else:
            self.spoken_frame = np.ones(shape=(self.spoken_POINTS)) + np.random.random((self.spoken_POINTS))*0.005
        if self.data is None:
            return
        if not self.active:
            WHITE = (117,251,251)
        else:
            WHITE = (117,251,251)
        frame = np.array(resample(self.data, self.POINTS))
        frame = frame.reshape((frame.shape[0])) + 1

        # GREEN
        c1 = self.circle.copy()
        c1[:, 0] *= frame
        c1[:, 1] *= frame
        c1 += 200

        # RED (TTS)
        c3 = self.spoken_circle.copy()
        c3[:, 0] *= self.spoken_frame
        c3[:, 1] *= self.spoken_frame
        c3 += 325


        # draw cv2
        # background
        cv2.polylines(self.cv2_img, np.int32([c1]), 1, WHITE[::-1], 3)
        cv2.polylines(self.cv2_img, np.int32([c3]), 1, (251,117,117)[::-1], 3)
        # blur
        self.cv2_img = cv2.GaussianBlur(self.cv2_img, (21, 21), 0)
        # highlight
        cv2.polylines(self.cv2_img, np.int32([c1]), 1, WHITE[::-1], 2)
        cv2.polylines(self.cv2_img, np.int32([c3]), 1, (251,117,117)[::-1], 2)
        self.cv2_img = cv2.GaussianBlur(self.cv2_img, (3, 3), 0)

        opencv_image = self.cv2_img[:,:,::-1]
        shape = opencv_image.shape[1::-1]
        pygame_image = pg.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
        rect = pygame_image.get_rect()
        rect.center = (200, 200)
        self.screen.blit(pygame_image, rect)
        self.cv2_img = np.zeros((400, 400, 3), np.uint8)
        return
    
    def update_data(self, data):
        self.data = data

class Handler:

    ### STT vars ###
    lang = "ru"
    name = ["atlas", "adlats", "ad las", "auf los", "at last", "ατλάσ"]
    API_KEYS = [0, "API-Key-1", "API-Key-2"]
    instructions = f"""
    You are an assistant called "Atlas" designed to help the user with various tasks. You do not use short forms when talking (like e. g., etc. and so on) and only use the same language as the last question asked.
    Instructions:
    - The only language you speak in is {lang}
    - You do not ask anything.
    - You do not use any abbreviations or short forms like e. g., etc. and so on.
    - You do not use special symbols like [], (), and :. The only allowed symbols are: .,-!?
    - Special symbols that need to be pronounced are typed out, so that any TTS-program is able to pronounce them.
    - Numbers are not represented as integers but as words, entirely typed out.
    - If the user asks you for the time, respond with an appropriate phrase and replace the actual values of the time with "[TIME hr]" and "[TIME min]". Do not say you don't know the time.
    - If the user asks for the date, respond with [DATE] and nothing else.
    - If the user asks you to switch on/off the lights, respond with [LIGHT ON/OFF] accordingly and do not say anything else.
    - use [START PC] if the user asks you to start the computer or start the setup
    - use [END PC] if the user asks you to shut down the pc or if he tells you that he is done
    - use [START BROWSER] if the user asks you to open a webbrowser
    - use [START CODE] if the user tells you he wants to start a programming project, continue with one or in general code something
    - if the user wants to open a website, use [WEB website] and replace "website" with the url of the requested website

    
    Context:
    - Your name is "Atlas" and you are a bot designed by "Iénissé". your task is to help the user.
    """
    messages = [{"role": "system", "content": instructions}]
    questions = 0
    key = 1
    openai.api_key = API_KEYS[key]

    ### TTS vars ###
    
    active = False

    commands = {
        "TIME hr": lambda: num2words(datetime.datetime.now().hour, lang="ru"),
        "TIME min": lambda: num2words(datetime.datetime.now().minute, lang="ru"),
        "LIGHT ON": lambda: print("Lights are being turned off!"),
        "START PC": lambda: os.system("vivaldi-stable && code"),
        "END PC": lambda: os.system("shutdown now"),
        "START BROWSER": lambda: os.system("vivaldi-stable"),
        "START CODE": lambda: os.system("code"),
        "WEB": lambda x: webbrowser.open(x),
    }

    def __init__(self, v=None, s=None) -> None:

        self.v = v
        self.s = s
        self.vad = webrtcvad.Vad(1)
        self.sample_rate = 16000
        self.min_speech_len = 200
        self.end = 15
        self.actual = -1
        self.speech = []
        self.model = whisper.load_model("medium")
    
    def update(self, recording):
        frame = resample(recording, int(self.sample_rate * DURATION / 1000))

        # convert to pcm16
        fr = (frame * 32768).astype(np.int16).tobytes()

        # check for speech
        contains_speech = self.vad.is_speech(fr, self.sample_rate)
        if contains_speech:
            self.speech += recording.tolist()
            self.actual = 0
            return
            
        if self.actual != -1:
            self.actual += 1
            self.speech += recording.tolist()
            if self.actual == self.end and len(self.speech) > self.min_speech_len:
                self.actual = -1
                speech_thread = threading.Thread(target=self.analyze_audio)
                speech_thread.start()
    
    def analyze_audio(self):
        if self.active:
            return False
        self.active = True
        audio = np.array(self.speech)
        self.speech = []
        audio = audio.reshape((audio.shape[0],)).astype(np.float32)
        wavfile.write('input.wav', FREQ, audio)
        audio = whisper.load_audio("input.wav")
        result = whisper.transcribe(self.model, audio)
        self.manage_request(result["text"])
    
    def check_name(self, msg):
        print("[RECOGNIZED (active: False)]", msg)
        msg = str(msg).lower()
        for i in self.name:
            if i in msg:
                return True
        return False

    def handle(self, msg):
        if self.questions == 3:
            self.key *= -1
            openai.api_key = self.API_KEYS[self.key]
            self.questions = 0
        self.messages.append({"role": "user", "content": msg})
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        self.messages.append(completion.choices[0].message)
        self.questions += 1
        return completion.choices[0].message.content.strip()
    
    def send_text(self, answer):
        print(f"[ANSWER (api_key: {self.key})]", answer)
        self.s.send(str(answer).encode())
        a = self.s.recv(1024).decode()

        self.v.spoken = np.load("out.npy")
        self.v.spoken_bool = True

        b = self.s.recv(1024)
        self.v.spoken_bool = False
        print("[ANSWER END]")
        self.active = False
        self.v.active = False
    
    def cmd(self, answer: list):
        if "[" in answer:
            answer = list(answer)
            for _ in range(answer.count("[")):
                start = answer.index("[")
                end = answer.index("]")
                cmd = "".join(answer[start+1:end])
                if "WEB" in cmd:
                    cmds = cmd.split()
                    self.commands[cmds[0]](cmds[1])
                    insert = ""
                else:
                    insert = str(self.commands[cmd]())
                del answer[start:end+1]
                answer.insert(start, insert)
            answer = "".join(answer)
        return answer

    def manage_request(self, msg):
        if self.v.active and "MBC" not in msg:
            print("[RECOGNIZED (active: True)]", msg)
            answer = self.cmd(self.handle(msg))
            if answer is not None and answer != 0:
                self.send_text(answer)
        elif self.check_name(msg):
            self.v.active = True
            self.active = False
            return
        
        else:
            self.active = False





def update(v,s):
    print("Initializing STT model...")
    handler = Handler(v, s)
    v.loaded = True
    while not v.done:
        recording = sd.rec(int(FREQ * DURATION / 1000), 
                        samplerate=FREQ, channels=1)
        sd.wait()

        v.update_data(recording)
        handler.update(recording)




if __name__ == "__main__":

    HOST = "127.0.0.1"  # The server's hostname or IP address
    PORT = 65432  # The port used by the server

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        FREQ = 44100
        DURATION = 30

        print("Initializing visualizer...")
        v = Visualizer()

        t = threading.Thread(target=update, args=(v,s,))
        t.start()

        v.start()

### TODO: compare this to temp.py -> why does this one not work?
