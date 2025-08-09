import time
import torch
import mss
import numpy as np
from PIL import Image
import easyocr
from neural_elis import ElisLSTM, N_LETTERS
from tokenizer import tokenize
from train_utils import train_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ElisLSTM(N_LETTERS, 512, N_LETTERS, num_layers=3).to(device) # количество нейронов и слоев LSTM ТЫКАТЬ ПОД ГП ОБЯЗАТЕЛЬНО ТЕ ЧТО В ЭЛИСЕ
MODEL_PATH = "models/elis_best.pt"
if torch.cuda.is_available():
    model.load_state_dict(torch.load(MODEL_PATH))
else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
reader = easyocr.Reader(['ru', 'en'], gpu=torch.cuda.is_available(), model_storage_directory='models/', user_network_directory='models/', recog_network='standard')

def grab_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Весь экран можно и на 2
        screenshot = np.array(sct.grab(monitor))
        image = Image.fromarray(screenshot)
        return image

def recognize_text(image):
    results = reader.readtext(np.array(image))
    lines = [text for (_, text, _) in results]
    return " ".join(lines)

def main():
    print("[Elis] Сбор с экрананачался...") # (Ctrl+C для остановки) возможно...
    try:
        while True:
            image = grab_screen()
            text = recognize_text(image)
            print(f"[SCREEN] {text}")

            if text:
                input_tensor = tokenize(text).to(device)
                if input_tensor.size(0) > 1: # хотя бы 2 символа а не бля и так робит
                    try:
                        loss = train_step(model, input_tensor, optimizer, loss_fn)
                        print(f"[TRAIN] Loss: {loss:.4f}")
                    except Exception as e:
                        print(f"[ERROR] Training step failed: {e}")

            time.sleep(1)# Пауза в секах брат можно больше если компу пизда
    except KeyboardInterrupt:
        print("\n[Elis] Остановлено пользователем.")

if __name__ == "__main__":
    main()
