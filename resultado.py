import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model

# 1. Carregar o modelo salvo
modelo_final = load_model('pipeline_precificacao_pycaret')

def prever_preco_imovel(metragem, quartos, banheiros, vagas, bairro, tipo):
    """
    Recebe os dados de um imóvel e retorna o valor estimado em Reais.
    """
    # Criar um DataFrame com os dados de entrada
    dados_entrada = pd.DataFrame([{
        'Metragem_Clean': metragem,
        'Quartos_Clean': quartos,
        'Banheiros_Clean': banheiros,
        'Vagas_Clean': vagas,
        'Bairro': bairro,
        'Tipo_Imovel': tipo
    }])

    # Fazer a predição (o resultado virá na coluna 'prediction_label' na escala Log)
    predicao_log = predict_model(modelo_final, data=dados_entrada)
    valor_log = predicao_log['prediction_label'].iloc[0]

    # Converter de Log para Reais (R$)
    valor_real = np.expm1(valor_log)
    
    return valor_real

# --- TESTE PRÁTICO ---
# Exemplo: Uma casa no Jardim Goiás de 150m²
estimativa = prever_preco_imovel(
    metragem=90, 
    quartos=3, 
    banheiros=1, 
    vagas=2, 
    bairro='Da Vitória', # Certifique-se que o nome do bairro existe no seu CSV
    tipo='Casa'
)

print(f"\nValor estimado pelo modelo: R$ {estimativa:,.2f}")