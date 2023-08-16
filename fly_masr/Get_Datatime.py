import os
import wave
import contextlib
import numpy as np
import cv2


def Get_labels(path):
# path定义要获取的文件名称的目录（C盘除外）
    # os.listdir()方法获取文件夹名字，返回数组
    file_name_list = os.listdir(path)
    # 转为转为字符串
    file_name = str(file_name_list)
    # replace替换"["、"]"、" "、"'"
    file_name = file_name.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n").replace(" ", "")
    # 创建并打开文件list.txt
    f = open("label_list.txt", "w")
    # 将文件下名称写入到"文件label_list.txt"
    f.write(file_name)
    f.close()
    labels = []
    f = open("label_list.txt", "r",encoding='utf-8')
    line = f.readline() # 读取第一行
    while line:
        line = line.strip()
        labels.append(line) # 列表增加
        line = f.readline() # 读取下一行
    f.close()
    return labels

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


def extract_features():
    data_path = "data/data_aishell/wav/test"
    labels = Get_labels(data_path)
    time = 0

    for label in labels:
        label_path = data_path + "\\" + label
        wav_names = os.listdir(label_path)
        for wav_name in wav_names:
            if wav_name.endswith(".wav"):
                wav_path = label_path + "\\" + wav_name
                time += get_duration_wav(wav_path)

    print(time)


if __name__ == '__main__':
    extract_features()
