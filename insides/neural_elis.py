import torch
import torch.nn as nn
import torch.optim as optim
import string
import os

# Расширенный алфавит с цифрами и арифметикой
ALL_LETTERS = (
    string.ascii_lowercase +
    "0123456789+-*/= .,;!?абвгдеёжзийклмнопрстуфхцчшщьыъэюя"
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

def load_partial_weights(old_model_path, new_model):
    old_state = torch.load(old_model_path, map_location=device)
    new_state = new_model.state_dict()

    loaded = 0
    for name, param in old_state.items():
        if name in new_state and param.shape == new_state[name].shape:
            new_state[name] = param
            loaded += 1
        else:
            print(f"Пропущено: {name} (размер {param.shape} → {new_state.get(name, 'не найден')})")

    new_model.load_state_dict(new_state)
    print(f"Загружено {loaded} совместимых параметров из старой модели.")
    
rnn = ElisLSTM(N_LETTERS, 256, N_LETTERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

MODEL_PATH = "models/elis_best.pt"
if os.path.exists(MODEL_PATH):
    try:
        load_partial_weights(MODEL_PATH, rnn)
        print(f"[Elis] Частично загружена модель из {MODEL_PATH}")
    except Exception as e:
        print(f"[Elis] Ошибка загрузки модели: {e}")