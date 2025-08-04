import time
import torch
import pyautogui
import torchaudio
from models import elis_best, elis_final
from pytesseract import pytesseract
from tokenizer import tokenize
from train_utils import train_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = elis_best(vocab_size=YOUR_VOCAB_SIZE).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = torch.nn.CrossEntropyLoss()

mic = torchaudio.io.StreamReader(src="default")
print("Сбор с экрана + микрофона начался...")

while True:
    try:
        screenshot = pyautogui.screenshot()
        text_data = pytesseract.image_to_string(screenshot)

        # Микрофон (опционально)
        # audio_data = mic.read(...)  распознавание через Whisper, если хочешь

        if text_data.strip():
            tokens = tokenize(text_data).to(device)
            loss = train_step(model, tokens, optimizer, loss_fn)
            print(f"[Обучение] Потеря: {loss:.4f}")
        
        time.sleep(1) 

    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
        torch.save(model.state_dict(), "elis_screen_trained.pt")
        break