import pyaudio
import numpy as np
from scipy.fftpack import fft
import wave
from scipy.io import wavfile
import noisereduce as nr
import time


def recording(filename, time1=0, threshold=8000):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = time1
    WAVE_OUTPUT_FILENAME = filename
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* 录音中...")
    frames = []
    if time1 > 0:
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data1 = stream.read(CHUNK)
            frames.append(data1)
    else:
        stopflag = 0
        stopflag2 = 0
        while True:
            data1 = stream.read(CHUNK)
            rt_data = np.frombuffer(data1, np.dtype('<i2'))
            # 傅里叶变换
            fft_temp_data = fft(rt_data, rt_data.size, overwrite_x=True)
            fft_data = np.abs(fft_temp_data)[0:fft_temp_data.size // 2 + 1]
            # 判断麦克风是否停止，判断说话是否结束，# 麦克风阈值，默认7000
            if sum(fft_data) // len(fft_data) > threshold:
                stopflag += 1
            else:
                stopflag2 += 1
            oneSecond = RATE / CHUNK
            print(stopflag2, " ", oneSecond)
            if stopflag2 + stopflag > oneSecond:
                if stopflag2 > oneSecond // 3 * 2:
                    break
                else:
                    stopflag2 = 0
                    stopflag = 0
            frames.append(data1)
    print("* 录音结束")
    stream.stop_stream()
    stream.close()
    p.terminate()
    print(p.get_sample_size(FORMAT))
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def noise_reduce():
    rate, data = wavfile.read(r'C:\Users\86138\Desktop\Chatbot\Audio\input.wav')
    _, noisy_part = wavfile.read(r'C:\Users\86138\Desktop\Chatbot\Audio\noise.wav')
    SAMPLING_FREQUENCY = 16000
    reduced_noise = nr.reduce_noise(y=data, y_noise=noisy_part, sr=SAMPLING_FREQUENCY)

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = time
    WAVE_OUTPUT_FILENAME = "C:\\Users\\86138\\Desktop\\Chatbot\\Audio\\output.wav"

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(reduced_noise))


if __name__ == '__main__':
    # recording('ppp.wav', time=2)  # 按照时间来录音，录音5秒
    recording('input.wav')  # 没有声音自动停止，自动停止
    recording('noise.wav')  # 没有声音自动停止，自动停止
    noise_reduce()

