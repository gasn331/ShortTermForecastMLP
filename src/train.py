import torch


def train_model(model, criterion, optimizer, X_train, y_train, save_dir='models'):

    print("Inicio do treinamento")

    # Treinar o modelo
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train.squeeze())

        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Exibir o progresso
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Salvar o modelo
    save_path = save_dir + '/trained_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Modelo salvo em: {save_path}")

    print("Fim do treinamento")
    return model
