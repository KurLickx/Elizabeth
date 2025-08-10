import torch
import torch.nn as nn
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
    tensor = torch.zeros(1, N_LETTERS, device=device)
    if letter in ALL_LETTERS:
        tensor[0][ALL_LETTERS.index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), N_LETTERS, device=device)
    for i, letter in enumerate(line):
        if letter in ALL_LETTERS:
            tensor[i][ALL_LETTERS.index(letter)] = 1
    return tensor

class ElisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(ElisLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )

def load_partial_weights(old_model_path, new_model):
    if not os.path.exists(old_model_path):
        print(f"[Elis] Файл модели не найден: {old_model_path}")
        return
    old_state = torch.load(old_model_path, map_location=device)
    new_state = new_model.state_dict()
    loaded = 0
    for name, param in old_state.items():
        if name in new_state and param.shape == new_state[name].shape:
            new_state[name] = param
            loaded += 1
        else:
            print(f"[Elis] Пропущен параметр: {name}")
    new_model.load_state_dict(new_state, strict=False)
    print(f"[Elis] Загружено {loaded} параметров из старой модели.")

rnn = ElisLSTM(N_LETTERS, 512, N_LETTERS, num_layers=3).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)