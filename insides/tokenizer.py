import torch
import string

ALL_LETTERS = (
    string.ascii_letters +
    "0123456789" +
    "+-*/= .,;!?_\\/№#@&()[]<>\"\'^$%~`|" +
    "абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"
)
N_LETTERS = len(ALL_LETTERS)

def tokenize(text):
    indices = []
    for letter in text:
        if letter in ALL_LETTERS:
            indices.append(ALL_LETTERS.index(letter))
    return torch.tensor(indices, dtype=torch.long)

def detokenize(indices):
    if torch.is_tensor(indices):
        indices = indices.tolist()
    return ''.join(ALL_LETTERS[i] for i in indices if 0 <= i < N_LETTERS)