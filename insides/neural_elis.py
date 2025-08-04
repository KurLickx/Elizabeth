import time
import torch
import pyautogui
import pytesseract
import string
from train_utils import train_step  
from neural_elis import ElisLSTM, ALL_LETTERS, N_LETTERS  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        if letter in ALL_LETTERS:
            tensor[i][0][ALL_LETTERS.index(letter)] = 1
    return tensor

model = ElisLSTM(N_LETTERS, 512, N_LETTERS, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

MODEL_PATH = "models/elis_final.pt"
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[Elis] Загружена финальная модель.")
    except Exception as e:
        print(f"[Elis] Ошибка загрузки модели: {e}")

print("[Elis] Начинаю обучение по экрану...")

try:
    while True:
        screenshot = pyautogui.screenshot()
        text_data = pytesseract.image_to_string(screenshot)

        if text_data.strip():
            tensor_input = line_to_tensor(text_data.strip()).to(device)
            target = tensor_input[1:].squeeze(1)
            input_seq = tensor_input[:-1]
            hidden = model.init_hidden()

            output, _ = model(input_seq, hidden)
            output = output.squeeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        time.sleep(1) 
except KeyboardInterrupt:
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[Elis] Дообученная модель сохранена в {MODEL_PATH}")