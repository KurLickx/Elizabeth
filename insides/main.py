import os
import datetime
import torch
import json
import random
from neural_elis import letter_to_tensor, line_to_tensor, ALL_LETTERS, device, rnn, criterion, optimizer

IDENTITY_FILE = "identity.txt"
MEMORY_FILE = "memory.json"
LOG_DIR = "logs"
INPUT_DIR = "input"
GENERATIONS_DIR = "generations"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GENERATIONS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def add_to_memory(тип, ввод, результат):
    memory = load_memory()
    memory.append({
        "тип": тип,
        "ввод": ввод,
        "результат": результат
    })
    save_memory(memory)

def load_identity():
    try:
        with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Я не знаю, кто я..."

def write_log(entry):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    time = datetime.datetime.now().strftime("%H:%M:%S")
    log_path = os.path.join(LOG_DIR, f"log_{date}.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time}] {entry}\n")

def speak(text):
    print(f"Elis {text}")
    write_log(f"Elis {text}")

def train(line):
    line_tensor = line_to_tensor(line).to(device)
    target_tensor = torch.tensor([
        ALL_LETTERS.index(c) for c in line.lower()[1:] if c in ALL_LETTERS
    ] + [ALL_LETTERS.index(" ")], dtype=torch.long).to(device)

    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    loss = 0

    for i in range(target_tensor.size(0)):
        output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
        loss += criterion(output.squeeze(0), target_tensor[i].unsqueeze(0))

    loss.backward()
    optimizer.step()
    return loss.item() / line_tensor.size(0)

def generate(start_char='', max_len=10000):
    if not start_char or start_char not in ALL_LETTERS:
        start_char = random.choice(ALL_LETTERS)

    input = letter_to_tensor(start_char).to(device)
    hidden = rnn.init_hidden()

    output_str = start_char
    for _ in range(max_len):
        output, hidden = rnn(input.unsqueeze(0), hidden)
        probs = torch.softmax(output[0][0], dim=0).detach().cpu()
        topi = torch.multinomial(probs, 1)[0].item()
        letter = ALL_LETTERS[topi]
        output_str += letter
        input = letter_to_tensor(letter).to(device)
        if letter == '=' and len(output_str) > 3:
            break

    return output_str

def main():
    speak("Утро")
    identity = load_identity()
    speak("Читаю свою личность...")
    speak(identity)
    speak("Я готова учиться.")

    input_file = os.path.join(INPUT_DIR, "sample1.txt")
    if not os.path.exists(input_file):
        speak(f"Файл для обучения не найден: {input_file}")
        return

    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()

    epochs = 1000
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        for line in lines:
            line = line.strip()
            if line:
                total_loss += train(line)
        avg_loss = total_loss / len(lines) if lines else 0
        speak(f"Epoch {epoch+1}/{epochs}: средняя потеря = {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(MODEL_DIR, "elis_best.pt")
            torch.save(rnn.state_dict(), best_model_path)
            speak(f"Сохранила лучшую модель с потерей {best_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            reply = generate(start_char=random.choice(" "))
            speak(f"Промежуточная генерация после эпохи {epoch+1}: {reply}")
            with open("generations_log.txt", "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch+1:03d}: {reply}\n")

    final_path = os.path.join(MODEL_DIR, "elis_final.pt")
    torch.save(rnn.state_dict(), final_path)
    speak("Сохранила финальную модель.")
    reply = generate(start_char='1')
    speak(f"Я сгенерировал: {reply}")
    add_to_memory("генерация", "1", reply)

if __name__ == "__main__":
    main()