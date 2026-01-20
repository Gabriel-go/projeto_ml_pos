import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Carregamento
file_path = r'e:\pos\10-Final\PROJETO\ml\imoveis_goiania_1768678610.csv'
print(f"Carregando dados de: {file_path}")
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Erro ao ler arquivo: {e}")
    exit()

# 2. Limpeza Básica
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

df['Preco_Clean'] = df['Preco'].apply(clean_price)
df['Metragem_Clean'] = df['Metragem'].apply(clean_area)
df['Quartos_Clean'] = df['Quartos'].apply(clean_numeric)
df['Banheiros_Clean'] = df['Banheiros'].apply(clean_numeric)
df['Vagas_Clean'] = df['Vagas'].apply(clean_numeric)

df = df.dropna(subset=['Preco_Clean', 'Metragem_Clean'])
df = df[df['Metragem_Clean'] > 0] # Remove zero area to avoid division by zero

# 3. Tratamento de Outliers (IQR) - Estratégia 3
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

# 4. Engenharia de Features - Estratégia 1
# Log do Target
df['Log_Preco'] = np.log1p(df['Preco_Clean'])

# Codificação de Bairro
le = LabelEncoder()
df['Bairro_Encoded'] = le.fit_transform(df['Bairro'].astype(str))

# 5. Divisão
features = ['Metragem_Clean', 'Quartos_Clean', 'Banheiros_Clean', 'Vagas_Clean', 'Bairro_Encoded', 'Bairro']
target = 'Log_Preco' # Treinando com Log

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Engineering: Preço m2 por Bairro (Target Encoding Contextual)
# Calculado apenas no treino para evitar Data Leakage
train_join = X_train.copy()
train_join['Preco_Real'] = np.expm1(y_train) # volta ao preço real
train_join['Preco_m2'] = train_join['Preco_Real'] / train_join['Metragem_Clean']

# Cria dicionário de média por bairro
bairro_m2_mean = train_join.groupby('Bairro')['Preco_m2'].mean().to_dict()
global_mean_m2 = train_join['Preco_m2'].mean()

# Mapeia para X_train e X_test (preenchendo desconhecidos com a média global)
X_train['Media_m2_Bairro'] = X_train['Bairro'].map(bairro_m2_mean).fillna(global_mean_m2)
X_test['Media_m2_Bairro'] = X_test['Bairro'].map(bairro_m2_mean).fillna(global_mean_m2)

# Seleção final de features
model_features = ['Metragem_Clean', 'Quartos_Clean', 'Banheiros_Clean', 'Vagas_Clean', 'Bairro_Encoded', 'Media_m2_Bairro']
X_train_final = X_train[model_features]
X_test_final = X_test[model_features]

# Escalonamento - Estratégia 3
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_final)

# 6. Modelagem e Avaliação - Estratégia 2
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, is_scaled=False):
    # Se for linear, usa dados escalonados
    X_train_use = X_train_scaled if is_scaled else X_tr
    X_test_use = X_test_scaled if is_scaled else X_te
    
    model.fit(X_train_use, y_tr)
    y_pred_log = model.predict(X_test_use)
    
    # Inverter log para métricas reais
    y_pred = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_te)
    
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    r2 = r2_score(y_test_real, y_pred)
    
    print(f"\n--- {name} ---")
    print(f"RMSE: R$ {rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    return model

# Regressão Linear
evaluate_model("Regressão Linear", LinearRegression(), X_train_final, y_train, X_test_final, y_test, is_scaled=True)

# Random Forest Tunada (Mais árvores, profundidade controlada)
rf_params = {'n_estimators': 500, 'max_depth': 15, 'random_state': 42}
rf = RandomForestRegressor(**rf_params)
rf_model = evaluate_model("Random Forest (Tuned)", rf, X_train_final, y_train, X_test_final, y_test)

# Gradient Boosting (Substituto robusto para XGBoost/LightGBM)
gb_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 42}
gb = GradientBoostingRegressor(**gb_params)
gb_model = evaluate_model("Gradient Boosting", gb, X_train_final, y_train, X_test_final, y_test)

# Importância das Variáveis (Gradient Boosting)
print("\n--- Importância das Variáveis (Gradient Boosting) ---")
importances = gb_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': model_features, 'Importance': importances}).sort_values('Importance', ascending=False)
print(feature_imp_df)
