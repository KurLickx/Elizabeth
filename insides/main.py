import os
import datetime
import torch
import random
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from neural_elis import letter_to_tensor, line_to_tensor, ALL_LETTERS, device, rnn, criterion, optimizer

IDENTITY_FILE = "identity.txt"
LOG_DIR = "logs"
INPUT_DIR = "input"
GENERATIONS_DIR = "generations"
MODEL_DIR = "models"
BEST_MODEL_NAME = "elis_best.pt"
FINAL_MODEL_NAME = "elis_final.pt"
VAL_MIN_THRESHOLD = 1e-6

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GENERATIONS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_identity():
    try:
        with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Я не знаю, хто я..."

def write_log(entry):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
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

    input_tensors, target_tensors, lengths = [], [], []
    for line in lines:
        input_seq = line[:-1]
        target_seq = line[1:]
        if len(input_seq) == 0 or len(target_seq) == 0:
            continue
        target_indices = []
        for c in target_seq:
            if c in ALL_LETTERS:
                target_indices.append(ALL_LETTERS.index(c))
            else:
                target_indices.append(-100)  # игнорируем непонятные символы
        if len(target_indices) != len(input_seq):
            continue
        input_tensor = line_to_tensor(input_seq)
        target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
        lengths.append(len(input_seq))
    if not input_tensors:
        return 0

    sorted_data = sorted(zip(lengths, input_tensors, target_tensors), key=lambda x: x[0], reverse=True)
    lengths, input_tensors, target_tensors = zip(*sorted_data)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    input_padded = pad_sequence(input_tensors)  
    target_padded = pad_sequence(target_tensors, padding_value=-100)  
    batch_size = input_padded.size(1)
    hidden = rnn.init_hidden(batch_size)
    packed_input = pack_padded_sequence(input_padded, lengths_tensor.cpu(), enforce_sorted=True)
    packed_output, hidden = rnn.lstm(packed_input, hidden)
    output, _ = pad_packed_sequence(packed_output)
    output = rnn.decoder(output)  # (seq_len, batch, output_size)
    output_flat = output.view(-1, output.size(-1))
    target_flat = target_padded.view(-1)

    loss = criterion(output_flat, target_flat)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=5)
    optimizer.step()

    return loss.item()

def validate(lines):
    if not lines:
        return 0
    total_loss = 0
    count = 0
    with torch.no_grad():
        for line in lines:
            line = line.strip()
            if len(line) < 2:
                continue
            try:
                input_seq = line[:-1]
                target_seq = line[1:]
                target_indices = []
                for c in target_seq:
                    if c in ALL_LETTERS:
                        target_indices.append(ALL_LETTERS.index(c))
                    else:
                        target_indices.append(-100)
                if len(target_indices) != len(input_seq):
                    continue
                input_tensor = line_to_tensor(input_seq).unsqueeze(1)
                target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
                hidden = rnn.init_hidden(batch_size=1)
                output, _ = rnn.lstm(input_tensor, hidden)
                output = rnn.decoder(output).squeeze(1)
                loss = criterion(output, target_tensor)
                total_loss += loss.item()
                count += 1
            except Exception as e:
                speak(f"Ошибка в валидации: {line[:30]}... Ошибка: {e}")
    return total_loss / count if count > 0 else 0

def generate(start_char='', max_len=100):
    if not start_char or start_char not in ALL_LETTERS:
        start_char = random.choice(ALL_LETTERS)
    input_tensor = letter_to_tensor(start_char)
    hidden = rnn.init_hidden(batch_size=1)
    output_str = start_char
    for _ in range(max_len):
        output, hidden = rnn(input_tensor.unsqueeze(0), hidden)
        probs = torch.softmax(output[0][0], dim=0).detach().cpu()
        topi = torch.multinomial(probs, 1)[0].item()
        letter = ALL_LETTERS[topi]
        output_str += letter
        input_tensor = letter_to_tensor(letter)
        if letter == '=' and len(output_str) > 3:
            break
    return output_str

def main():
    speak("Утро")
    speak(load_identity())
    speak("Я готова учиться.")

    input_file = os.path.join(INPUT_DIR, "sample1.txt")
    if not os.path.exists(input_file):
        speak(f"Файл для обучения не найден: {input_file}")
        return

    with open(input_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    epochs = 500 # Количество эпох 
    batch_size = 64    # Размер батча
    val_ratio = 0.1
    best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    if os.path.exists(best_model_path):
        try:
            rnn.load_state_dict(torch.load(best_model_path, map_location=device))
            speak(f"Загружена модель из {best_model_path}")
        except Exception as e:
            speak(f"Ошибка загрузки модели: {e}")

    best_val_loss = float("inf")
    for epoch in range(epochs):
        random.shuffle(lines)
        val_size = int(len(lines) * val_ratio)
        val_lines = lines[:val_size]
        train_lines = lines[val_size:]
        total_loss = 0
        count_batches = 0
        for i in range(0, len(train_lines), batch_size):
            batch = train_lines[i:i + batch_size]
            loss = train_batch(batch)
            if loss == 0:
                continue
            total_loss += loss
            count_batches += 1
        avg_train_loss = total_loss / count_batches if count_batches > 0 else 0
        val_loss = validate(val_lines)
        scheduler.step(val_loss)

        speak(f"Эпоха {epoch+1}/{epochs} — Трен. потери: {avg_train_loss:.6f}, Валидация: {val_loss:.6f}")
        if val_loss < best_val_loss - VAL_MIN_THRESHOLD:
            best_val_loss = val_loss
            torch.save(rnn.state_dict(), best_model_path)
            speak(f"Модель улучшилась! Сохраняю в {best_model_path}")
        if (epoch + 1) % 50 == 0:
            sample = generate('=')
            speak(f"Пример генерации: {sample}")
    final_path = os.path.join(MODEL_DIR, FINAL_MODEL_NAME)
    torch.save(rnn.state_dict(), final_path)
    speak(f"Обучение завершено, модель сохранена в {final_path}")

if __name__ == "__main__":
    main()
