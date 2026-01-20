# Roteiro do Pipeline de Machine Learning: Precificação de Imóveis

Este documento detalha o fluxo completo de Ciência de Dados (Data Science Pipeline) que será implementado para criar o modelo preditivo de preços de imóveis em Rio Verde.

## 1. Coleta e Carregamento de Dados (Data Ingestion)
*   **Objetivo**: Carregar os dados brutos do arquivo `imoveis_rio_verde.csv` para um formato manipulável (DataFrame).
*   **Ação**: Ler o CSV, verificando encoding e separadores.

## 2. Limpeza e Pré-processamento (Data Cleaning & Preprocessing)
A etapa mais crítica, dado o formato "sujo" (strings) dos dados numéricos.
*   **Tratamento de Preço**:
    *   Remover caracteres "R$", pontos e espaços.
    *   Converter para tipo numérico (float).
*   **Tratamento de Metragem**:
    *   Remover " m²" e outros textos.
    *   Converter para numérico.
*   **Tratamento de Variáveis Númericas (Quartos, Banheiros, Vagas)**:
    *   Identificar valores "N/A".
    *   Definir estratégia para nulos: Substituir por 0 (assumindo inexistência) ou Mediana/Média? (Para terrenos, quartos será 0).
    *   Converter para inteiro/float.
*   **Tratamento de Texto (Bairro/Cidade)**:
    *   Padronizar nomes de bairros e cidade.

## 3. Análise Exploratória de Dados (EDA)
*   **Objetivo**: Entender o comportamento dos dados antes de modelar.
*   **Ações**:
    *   Análise de distribuição do Target (`Preco`).
    *   Identificação de Outliers (imóveis excessivamente caros ou baratos).
    *   Correlação entre variáveis (ex: Metragem vs Preço).
    *   Visualização da média de preço por Bairro.

## 4. Engenharia de Atributos (Feature Engineering)
*   **Codificação de Variáveis Categóricas**:
    *   Transformer a coluna `Bairro` em números (One-Hot Encoding ou Target Encoding).
*   **Criação de Novas Features**:
    *   Extrair tipo de imóvel do título (Casa, Apartamento, Lote).

## 5. Divisão do Dataset (Train-Test Split)
*   **Objetivo**: Garantir que o modelo seja testado em dados inéditos.
*   **Ação**: Separar os dados em Treino (80%) e Teste (20%).

## 6. Modelagem (Modeling)
Serão testados algoritmos de regressão.
*   **Baseline**: Regressão Linear.
*   **Modelos Avançados**: Random Forest Regressor e/ou Gradient Boosting.

## 7. Avaliação e Métricas (Evaluation)
*   **Métricas**: RMSE (Raiz do Erro Quadrático Médio) e R² (Coeficiente de Determinação).

## 8. Entrega (Deployment)
*   Criar script de inferência para prever preços de novos exemplos.
