import pandas as pd
import numpy as np
from pycaret.regression import *
import re

# 1. Carregamento
file_path = r'e:\pos\10-Final\PROJETO\ml\imoveis_goiania.csv'
#file_path = r'e:\pos\10-Final\PROJETO\ml\imoveis_rio_verde.csv'
print(f"Carregando dados de: {file_path}")
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Erro ao ler arquivo: {e}")
    exit()

# 2. Limpeza Básica
def extract_property_type(title):
    if pd.isna(title): return None
    match = re.match(r'(\w+)\s+para comprar', title, re.IGNORECASE)
    if match:
        return match.group(1)
    return 'Outros'

def clean_price(price_str):
    if pd.isna(price_str): return np.nan
    clean_str = str(price_str).replace('R$', '').replace('.', '').replace(' ', '').strip()
    try: return float(clean_str)
    except: return np.nan

def clean_area(area_str):
    if pd.isna(area_str): return np.nan
    clean_str = str(area_str).lower().replace('m²', '').replace(' ', '').strip()
    try: return float(clean_str)
    except: return np.nan

def clean_numeric(val):
    if pd.isna(val) or str(val).strip() == 'N/A': return 0
    try: return float(val)
    except: return 0
df['Tipo_Imovel'] = df['Titulo'].apply(extract_property_type)
df['Preco_Clean'] = df['Preco'].apply(clean_price)
df['Metragem_Clean'] = df['Metragem'].apply(clean_area)
df['Quartos_Clean'] = df['Quartos'].apply(clean_numeric)
df['Banheiros_Clean'] = df['Banheiros'].apply(clean_numeric)
df['Vagas_Clean'] = df['Vagas'].apply(clean_numeric)

df = df.dropna(subset=['Preco_Clean', 'Metragem_Clean'])
df = df[df['Tipo_Imovel'] != 'Outros']


# 3. Tratamento de Outliers (IQR)
print("\n--- Tratamento de Outliers ---")
print(f"Registros antes da remoção: {len(df)}")

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'Preco_Clean')
df = remove_outliers(df, 'Metragem_Clean')
print(f"Registros após remoção (IQR em Preço e Metragem): {len(df)}")

# 4. Feature Engineering Manual (Log Target)
# Evitando problemas com transform_target do PyCaret (scipy optimize error)
df['Log_Preco'] = np.log1p(df['Preco_Clean'])

# 5. PyCaret Setup
print("\n--- Iniciando Setup do PyCaret ---")
# Features para usar
features = ['Metragem_Clean', 'Quartos_Clean', 'Banheiros_Clean', 'Vagas_Clean', 'Bairro', 'Log_Preco','Tipo_Imovel']
# Filtrando apenas colunas úteis
data_ml = df[features].copy()

# O setup do PyCaret
s = setup(data=data_ml, 
          target='Log_Preco', 
          train_size=0.8,
          categorical_features=['Bairro'], 
          numeric_features=['Metragem_Clean', 'Quartos_Clean', 'Banheiros_Clean', 'Vagas_Clean'],
          transform_target=False, # Processamento manual do alvo
          session_id=42,
          verbose=True)

# 6. Comparação de Modelos
print("\n--- Comparando Modelos (Target Log Transformado) ---")
# Compara models. Note que o erro será na escala Log.
best_model = compare_models(sort='RMSE')

print("\nMelhor Modelo Encontrado:")
print(best_model)

# 7. Finalização e Salvamento
print("\n--- Finalizando e Salvando Modelo ---")
final_model = finalize_model(best_model)
save_model(final_model, 'pipeline_precificacao_pycaret')
print("Modelo salvo como 'pipeline_precificacao_pycaret.pkl'")
