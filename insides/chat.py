import torch
import string
from neural_elis import ElisLSTM

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

def char_to_onehot(index):
    onehot = torch.zeros(1, 1, N_LETTERS, dtype=torch.float32) 
    onehot[0, 0, index] = 1.0
    return onehot

def generate(model, start_char, max_length=100): #тут длинна её предложения (максимальная)
    model.eval()
    hidden = model.init_hidden(batch_size=1)
    input_idx = tokenize(start_char)[0].item()
    input_tensor = char_to_onehot(input_idx)
    output_str = start_char

    for _ in range(max_length - 1):
        output, hidden = model(input_tensor, hidden)
        probs = torch.softmax(output.view(-1), dim=0)
        top_idx = torch.multinomial(probs, 1)[0].item()
        next_char = detokenize([top_idx])
        output_str += next_char
        input_tensor = char_to_onehot(top_idx)

    return output_str

state_dict = torch.load("models/elis_best.pt", map_location="CUDA" if torch.cuda.is_available() else "cpu")
hidden_size = state_dict['lstm.weight_ih_l0'].size(0)//4
num_layers = len([k for k in state_dict if k.startswith("lstm.weight_ih_l")])

model = ElisLSTM(
    input_size=N_LETTERS,
    hidden_size=hidden_size,
    output_size=N_LETTERS,
    num_layers=num_layers
)

model.load_state_dict(state_dict)

print("Elis чат готов.")

while True:
    user_input = input("Я: ")
    if user_input.lower() == "exit":
        break
    if not user_input.strip():
        continue
    start_char = user_input[0]
    response = generate(model, start_char=start_char, max_length=100) #тут длинна твоего предложения (максимальная)
    print(f"Elis: {response}")