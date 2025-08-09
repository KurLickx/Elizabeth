import torch
import string

ALL_LETTERS = (
    string.ascii_letters +
    "0123456789" +
    "+-*/= .,;!?_\\/№#@&()[]<>\"\'^$%~`|" +
    "абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"
)
N_LETTERS = len(ALL_LETTERS)
letter_to_index = {ch: i for i, ch in enumerate(ALL_LETTERS)}

def tokenize(text):
    indices = [letter_to_index[ch] for ch in text if ch in letter_to_index]
    return torch.tensor(indices, dtype=torch.long)

def detokenize(indices):
    if torch.is_tensor(indices):
        indices = indices.tolist()
    return ''.join(ALL_LETTERS[i] for i in indices if 0 <= i < N_LETTERS)