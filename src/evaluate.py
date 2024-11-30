def evaluate_model(model, criterion, X_test, y_test):
    model.eval()
    y_pred = model(X_test)

    # Calcular o erro médio quadrático
    mse = criterion(y_pred.squeeze(), y_test.squeeze())
    print(f'Mean Squared Error (MSE) no conjunto de teste: {mse.item():.4f}')

    return y_pred
