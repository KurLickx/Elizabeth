import os
import datetime
import torch
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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
    lines = [line.strip() for line in batch_lines if len(line.strip()) > 1]
    if not lines:
        return 0

    input_tensors = []
    target_tensors = []
    lengths = []

    for line in lines:
        input_seq = line[:-1]
        target_seq = line[1:]
        input_tensor = line_to_tensor(input_seq).to(device)
        target_indices = [ALL_LETTERS.index(c) for c in target_seq if c in ALL_LETTERS]
        if len(target_indices) != len(input_seq):
            continue
        input_tensors.append(input_tensor)
        target_tensors.append(torch.tensor(target_indices, dtype=torch.long, device=device))
        lengths.append(len(input_seq))

    if not input_tensors:
        return 0

    input_padded = pad_sequence(input_tensors)
    target_padded = pad_sequence(target_tensors, padding_value=-100)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    batch_size = input_padded.size(1)
    hidden = (
        torch.zeros(rnn.num_layers, batch_size, rnn.hidden_size, device=device),
        torch.zeros(rnn.num_layers, batch_size, rnn.hidden_size, device=device),
    )
    packed_input = pack_padded_sequence(input_padded, lengths_tensor.cpu(), enforce_sorted=False)
    packed_output, hidden = rnn.lstm(packed_input, hidden)
    output, _ = pad_packed_sequence(packed_output)
    output = rnn.decoder(output)
    output_flat = output.view(-1, output.size(-1))
    target_flat = target_padded.view(-1)
    loss = criterion(output_flat, target_flat)
    loss.backward()
    optimizer.step()
    return loss.item()

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
                input_seq = line[:-1]
                target_seq = line[1:]
                input_tensor = line_to_tensor(input_seq).to(device)
                target_indices = [ALL_LETTERS.index(c) for c in target_seq if c in ALL_LETTERS]
                if len(target_indices) != len(input_seq):
                    continue  
                target_tensor = torch.tensor(target_indices, dtype=torch.long).to(device)
                input_tensor = input_tensor.unsqueeze(1)
                hidden = (torch.zeros(rnn.num_layers, 1, rnn.hidden_size, device=device),
                          torch.zeros(rnn.num_layers, 1, rnn.hidden_size, device=device))
                output, _ = rnn.lstm(input_tensor, hidden)
                output = rnn.decoder(output)
                output = output.squeeze(1)
                loss = criterion(output, target_tensor)
                total_loss += loss.item()
            except Exception as e:
                speak(f"Ошибка в валидации строки: {line[:30]}... Ошибка: {e}")
    return total_loss / val_size if val_size > 0 else 0

def generate(start_char='', max_len=100):
    if not start_char or start_char not in ALL_LETTERS:
        start_char = random.choice(ALL_LETTERS)

    input_tensor = letter_to_tensor(start_char).to(device)
    hidden = rnn.init_hidden(batch_size=1)

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

    epochs = 1    #
    batch_size = 512 #
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

        if (epoch + 1) % 25 == 0:
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