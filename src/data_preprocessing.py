import pandas as pd


def load_data(filepath='data/GOIANIA_01-01-2015_A_29-08-2024.csv'):
    # Carregar o CSV
    df = pd.read_csv(filepath, encoding='ISO-8859-1', delimiter=';', decimal=',')

    # Selecionar as colunas que serão usadas como features
    features = [
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
        'PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)', 'PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)',
        'RADIACAO GLOBAL (Kj/m²)', 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
        'TEMPERATURA DO PONTO DE ORVALHO (°C)', 'TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)',
        'TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)', 'TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)',
        'TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)', 'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)',
        'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)', 'UMIDADE RELATIVA DO AR, HORARIA (%)',
        'VENTO, DIREÇÃO HORARIA (gr) (° (gr))', 'VENTO, RAJADA MAXIMA (m/s)', 'VENTO, VELOCIDADE HORARIA (m/s)'
    ]

    # Selecionar a variável target (a temperatura média do próximo dia)
    target = ['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)']  # Ou a temperatura para cada hora

    # Remover colunas que não são úteis ou são duplicadas
    df = df.dropna(subset=features + target)

    # Separar features e target
    X = df[features].values
    y = df[target].values
    dates = df['Data']  # Atribuindo as datas aos dados
    hours = df['Hora UTC']  # Atribuindo as horas

    return X, y, dates, hours
