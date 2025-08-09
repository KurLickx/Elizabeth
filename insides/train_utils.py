import torch
#это для обучения модели тоже не надо трогать
def train_step(model, input_tensor, optimizer, loss_fn):
    model.train()
    hidden = model.init_hidden()

    optimizer.zero_grad()
    output, hidden = model(input_tensor, hidden)
    output = output.squeeze(1)
    target = input_tensor[1:].squeeze(1).argmax(dim=1)
    prediction = output[:-1]

    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()

    return loss.item()