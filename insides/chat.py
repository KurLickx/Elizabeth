import torch
from neural_elis import ElisLSTM, N_LETTERS
import os
from tokenizer import tokenize, detokenize 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = N_LETTERS
model = ElisLSTM(vocab_size, 512, vocab_size).to(device)
model.load_state_dict(torch.load("elis_model.pt", map_location=device))
model.to(device)
model.eval()

print("Elis чат готов. Пиши что угодно:\n")

with torch.no_grad():
    while True:
        user_input = input("Я: ")
        if user_input.lower() in ("выход", "exit", "quit"):
            break
        input_tensor = tokenize(user_input).unsqueeze(0).to(device)
        output = model(input_tensor)
        response = detokenize(output.squeeze(0))
        print(f"Elis: {response}")