import torch
import torch.nn as nn
import torch.optim as optim
import string
import os

ALL_LETTERS = (
    string.ascii_letters + 
    "0123456789" +
    "+`-*/= .,;!?_\\/№#@&()[]<>\"\'^$%~|" +
    "абвгдеёжзийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ"  
)

N_LETTERS = len(ALL_LETTERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    if letter in ALL_LETTERS:
        tensor[0][ALL_LETTERS.index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), N_LETTERS)
    for i, letter in enumerate(line):
        if letter in ALL_LETTERS:
            tensor[i][ALL_LETTERS.index(letter)] = 1
    return tensor

class ElisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(ElisLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
            torch.zeros(self.num_layers, 1, self.hidden_size, device=device),
        )

def load_partial_weights(old_model_path, new_model):
    old_state = torch.load(old_model_path, map_location=device)
    new_state = new_model.state_dict()

    loaded = 0
    for name, param in old_state.items():
        if name in new_state:
            if param.shape == new_state[name].shape:
                new_state[name] = param
                loaded += 1
            else:
                print(f"[Elis] Пропущен параметр: {name} (размер {param.shape} → {new_state[name].shape})")
        else:
            print(f"[Elis] Пропущен параметр: {name} (не найден в новой модели)")

    new_model.load_state_dict(new_state, strict=False)
    print(f"[Elis] Загружено {loaded} совместимых параметров из старой модели.")

rnn = ElisLSTM(N_LETTERS, 512, N_LETTERS, num_layers=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)

MODEL_PATH = "models/elis_best.pt"
if os.path.exists(MODEL_PATH):
    try:
        load_partial_weights(MODEL_PATH, rnn)
        print(f"[Elis] Частично загружена модель из {MODEL_PATH}")
    except Exception as e:
        print(f"[Elis] Ошибка загрузки модели: {e}")