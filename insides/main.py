import os
import datetime
import torch
import random
from neural_elis import letter_to_tensor, line_to_tensor, ALL_LETTERS, device, rnn, criterion, optimizer

IDENTITY_FILE = "identity.txt"
LOG_DIR = "logs"
INPUT_DIR = "input"
GENERATIONS_DIR = "generations"
MODEL_DIR = "models"
BEST_MODEL_NAME = "elis_best.pt"
FINAL_MODEL_NAME = "elis_final.pt"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GENERATIONS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
    print(f"Elis: {text}")
    write_log(f"Elis: {text}")

def train_batch(batch_lines):
    optimizer.zero_grad()
    total_loss = 0
    batch_size = len(batch_lines)

    for line in batch_lines:
        line = line.strip()
        if len(line) < 2:
            continue
        try:
            line_tensor = line_to_tensor(line).to(device)
            target_indices = [ALL_LETTERS.index(c) for c in line[1:] if c in ALL_LETTERS]
            target_indices.append(ALL_LETTERS.index(" "))
            target_tensor = torch.tensor(target_indices, dtype=torch.long).to(device)

            hidden = rnn.init_hidden()
            loss = 0
            for i in range(target_tensor.size(0)):
                output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
                loss += criterion(output.squeeze(0), target_tensor[i].unsqueeze(0))
            loss.backward()
            total_loss += loss.item() / target_tensor.size(0)

        except Exception as e:
            speak(f"Ошибка при обучении строки: {line[:30]}... Ошибка: {e}")

    optimizer.step()
    return total_loss / batch_size if batch_size > 0 else 0

def validate(lines, val_ratio=0.1):
    val_size = max(1, int(len(lines)*val_ratio))
    val_lines = lines[:val_size]
    total_loss = 0
    with torch.no_grad():
        for line in val_lines:
            line = line.strip()
            if len(line) < 2:
                continue
            try:
                line_tensor = line_to_tensor(line).to(device)
                target_indices = [ALL_LETTERS.index(c) for c in line[1:] if c in ALL_LETTERS]
                target_indices.append(ALL_LETTERS.index(" "))
                target_tensor = torch.tensor(target_indices, dtype=torch.long).to(device)

                hidden = rnn.init_hidden()
                loss = 0
                for i in range(target_tensor.size(0)):
                    output, hidden = rnn(line_tensor[i].unsqueeze(0), hidden)
                    loss += criterion(output.squeeze(0), target_tensor[i].unsqueeze(0))
                total_loss += loss.item() / target_tensor.size(0)
            except Exception as e:
                speak(f"Ошибка в валидации строки: {line[:30]}... Ошибка: {e}")
    return total_loss / val_size if val_size > 0 else 0

def generate(start_char='', max_len=100):
    if not start_char or start_char not in ALL_LETTERS:
        start_char = random.choice(ALL_LETTERS)

    input_tensor = letter_to_tensor(start_char).to(device)
    hidden = rnn.init_hidden()

    output_str = start_char
    for _ in range(max_len):
        output, hidden = rnn(input_tensor.unsqueeze(0), hidden)
        probs = torch.softmax(output[0][0], dim=0).detach().cpu()
        topi = torch.multinomial(probs, 1)[0].item()
        letter = ALL_LETTERS[topi]
        output_str += letter
        input_tensor = letter_to_tensor(letter).to(device)
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
        lines = [line.strip() for line in f if line.strip()]

    epochs = 50
    batch_size = 16
    val_ratio = 0.1

    best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
    if os.path.exists(best_model_path):
        try:
            rnn.load_state_dict(torch.load(best_model_path, map_location=device))
            speak(f"Загружена модель из {best_model_path}")
        except Exception as e:
            speak(f"Ошибка загрузки модели: {e}")

    best_loss = float("inf")

    for epoch in range(epochs):
        random.shuffle(lines)
        train_lines = lines[int(len(lines)*val_ratio):]
        val_lines = lines[:int(len(lines)*val_ratio)]

        total_loss = 0
        for i in range(0, len(train_lines), batch_size):
            batch = train_lines[i:i+batch_size]
            loss = train_batch(batch)
            total_loss += loss * len(batch)

        avg_train_loss = total_loss / len(train_lines) if train_lines else 0
        avg_val_loss = validate(val_lines, val_ratio=0)  

        speak(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.6f} - Val loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(rnn.state_dict(), best_model_path)
            speak(f"Сохранила лучшую модель с потерей {best_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            reply = generate(start_char=random.choice(ALL_LETTERS))
            speak(f"Промежуточная генерация после эпохи {epoch+1}: {reply}")
            with open(os.path.join(GENERATIONS_DIR, "generations_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch+1:03d}: {reply}\n")

    final_path = os.path.join(MODEL_DIR, FINAL_MODEL_NAME)
    torch.save(rnn.state_dict(), final_path)
    speak("Сохранила финальную модель.")
    reply = generate(start_char='')
    speak(f"Я сгенерировал: {reply}")

if __name__ == "__main__":
    main()