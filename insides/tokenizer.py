import torch
import string

ALL_LETTERS = (
    string.ascii_letters +
    "0123456789" +
    "+-*/= .,;!?_\\/№#@&()[]<>\"\'^$%~|`" +
    "абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
N_LETTERS = len(ALL_LETTERS)

def tokenize(text):
    tensor = torch.zeros(len(text), 1, N_LETTERS)
    for i, letter in enumerate(text):
        if letter in ALL_LETTERS:
            index = ALL_LETTERS.index(letter)
            tensor[i][0][index] = 1
    return tensor