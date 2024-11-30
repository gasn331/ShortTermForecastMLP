from sklearn.preprocessing import StandardScaler
import torch
from src.mlp import MLP
from src.data_preprocessing import load_data
from src.predict_utils import predict_next_day, predict_day_after_final_data


# Função principal
def main():
    # Carregar os dados
    X, y, dates, hours = load_data('../data/GOIANIA_01-01-2015_A_29-08-2024.csv')

    # Normalizar as features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Converter para tensores PyTorch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Dividir os dados em treino e teste
    train_size = int(0.8 * len(X_tensor))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    # Criar o modelo MLP
    model = MLP(X_train.shape[1])

    # Carregar o modelo
    model_weights = torch.load('../models/trained_model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(model_weights)

    with torch.no_grad():
        predict_day_after_final_data(model=model, dates=dates, X_test=X_test, y_test=y_test)
        predict_next_day(model=model, X_test=X_test)


# Chamar a função principal
if __name__ == "__main__":
    main()
