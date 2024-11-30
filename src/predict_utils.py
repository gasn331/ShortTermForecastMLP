from datetime import datetime, timedelta
import torch


def predict_day_after_final_data(model, dates, X_test, y_test):
    # Prever para o próximo dia, hora a hora
    last_date = datetime.strptime(dates.iloc[-1], '%d/%m/%Y')  # A data do último valor
    next_day = last_date + timedelta(days=1)
    print(f'\nPrevisão para o dia seguinte ao final do conjunto de dados ({next_day.strftime("%d/%m/%Y")}):')

    # Criar uma lista para armazenar as previsões e reais
    previsoes = []
    reais = []

    # Prever a temperatura para cada hora do próximo dia
    X_next_day = X_test[-24:]  # Últimos 24 valores de teste (últimas 24 horas)
    for i in range(24):
        next_hour_pred = model(
            torch.tensor(X_next_day[i:i + 1], dtype=torch.float32).clone().detach()).item()  # Passando a linha i como entrada
        next_hour_time = (next_day + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M")

        # Adiciona as previsões e reais
        previsoes.append(next_hour_pred)
        reais.append(y_test[i].item())  # Usando o valor real para comparação

        print(f'{next_hour_time} | {next_hour_pred:.2f} °C')


def predict_next_day(model, X_test):
    # Obter a data e hora atuais
    current_datetime = datetime.now()
    print(f'\nData e hora atual: {current_datetime.strftime("%d/%m/%Y %H:%M")}')

    # Calcular a data e hora para o próximo dia
    next_day = current_datetime + timedelta(days=1)
    print(f'Início das previsões para o próximo dia ({next_day.strftime("%d/%m/%Y")}):')

    # Prever a temperatura para as próximas 24 horas
    previsoes = []
    for i in range(24):
        next_hour_time = (next_day + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M")
        # Simulando a entrada para previsão (substitua X_test[i] por sua entrada real, se necessário)
        next_hour_pred = model(
            torch.tensor(X_test[i:i + 1], dtype=torch.float32).clone().detach()).item()  # Prevendo para a próxima hora

        previsoes.append(next_hour_pred)
        print(f'{next_hour_time} | {next_hour_pred:.2f} °C')

    """# Mostrar as previsões para o próximo dia
    print(f'Previsões para o próximo dia ({next_day.strftime("%d/%m/%Y")}):')
    for i in range(24):
        next_hour_time = (next_day + timedelta(hours=i)).strftime("%d/%m/%Y %H:%M")
        print(f'{next_hour_time} | {previsoes[i]:.2f} °C')"""
