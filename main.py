from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from src.mlp import MLP
from src.data_preprocessing import load_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict_utils import predict_next_day, predict_day_after_final_data


# Função principal
def main():
    # Carregar os dados
    X, y, dates, hours = load_data()

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
    dates_test = dates[train_size:]  # Separar as datas para o teste
    hours_test = hours[train_size:]  # Separar as horas para o teste

    # Criar o modelo MLP
    model = MLP(X_train.shape[1])

    # Definir o critério de perda e o otimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_model(model=model, criterion=criterion, optimizer=optimizer, X_train=X_train, y_train=y_train)

    with torch.no_grad():
        # Avaliar o modelo
        y_pred = evaluate_model(model=model, criterion=criterion, X_test=X_test, y_test=y_test)

        # Exibir as previsões e os valores reais com as datas e horas
        print(f'\nData | Hora | Previsões | Reais')
        for date, hour, pred, real in zip(dates_test[-24:], hours_test[-24:], y_pred.squeeze().numpy()[-24:],
                                          y_test.squeeze().numpy()[-24:]):
            print(f'{date} {hour} | {pred:.2f} | {real:.2f}')

        predict_day_after_final_data(model=model, dates=dates, X_test=X_test, y_test=y_test)
        predict_next_day(model=model, X_test=X_test)


# Chamar a função principal
if __name__ == "__main__":
    main()
