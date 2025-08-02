import torch
import torch.nn as nn
import torch.optim as optim
import string
import os

ALL_LETTERS = (
    string.ascii_lowercase +
    "0123456789+-*/= .,;!?абвгдеёжзийклмнопрстуфхцчшщьыъэюя" +
    "_\\/№#@[]{}()<>^$&%~`|\"'"
)
N_LETTERS = len(ALL_LETTERS)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    if letter in ALL_LETTERS:
        tensor[0][ALL_LETTERS.index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line.lower()):
        if letter in ALL_LETTERS:
            tensor[i][0][ALL_LETTERS.index(letter)] = 1
    return tensor

class ElisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElisLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_partial_weights_extend(old_model_path, new_model):
    old_state = torch.load(old_model_path, map_location=device)
    new_state = new_model.state_dict()

    for name in new_state:
        if name in old_state:
            old_param = old_state[name]
            new_param = new_state[name]

            if old_param.shape == new_param.shape:
                new_state[name] = old_param
            elif name.startswith("decoder.weight") or name.startswith("decoder.bias"):
                min_shape = min(old_param.shape[0], new_param.shape[0])
                if "weight" in name:
                    new_state[name][:min_shape, :] = old_param[:min_shape, :]
                else:  # ура bias робит
                    new_state[name][:min_shape] = old_param[:min_shape]
                print(f"[Elis] Расширен слой {name} ({old_param.shape} → {new_param.shape})")
            else:
                print(f"[Elis] Пропущен параметр: {name} ({old_param.shape} → {new_param.shape})")

    new_model.load_state_dict(new_state)
    print("[Elis] Частично загружены совместимые веса модели.")

rnn = ElisLSTM(N_LETTERS, 512, N_LETTERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

MODEL_PATH = "models/elis_best.pt"
if os.path.exists(MODEL_PATH):
    try:
        load_partial_weights_extend(MODEL_PATH, rnn)
    except Exception as e:
        print(f"[Elis] Ошибка загрузки модели: {e}")