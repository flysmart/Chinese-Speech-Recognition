import os
import wave
import contextlib
import time



def get_duration_wav(file_path):
    """
    获取wav音频文件时长
    :param file_path:
    :return:
    """
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

