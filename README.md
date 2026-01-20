# 🏠 Precificação de Imóveis - Goiânia (ML Pipeline)

Este projeto implementa um pipeline completo de Machine Learning para predição de preços de imóveis em Goiânia/GO. O sistema utiliza técnicas avançadas de AutoML (via **PyCaret**) e análise estatística para estimar valores de mercado com alta precisão.

## 📊 Resultados do Modelo

O modelo final (**LightGBM**) apresentou performance excepcional para o mercado imobiliário:

| Métrica | Valor | Interpretação |
|:---:|:---:|:---|
| **R²** | **0.8340** | O modelo explica 83.4% da variação de preços. Acima de 0.80 é considerado excelente para precificação imobiliária. |
| **MAE** | **0.1706** | Erro médio absoluto na escala logarítmica. Indica um erro médio aproximado de **17%** sobre o valor do imóvel. |
| **RMSE** | **0.2312** | Proximidade com o MAE indica estabilidade e robustez contra outliers (preços absurdos). |

---

## 🛠️ Tecnologias Utilizadas

*   **Python 3.8+**
*   **PyCaret**: AutoML para treinamento e comparação de modelos.
*   **Pandas & NumPy**: Manipulação e limpeza de dados.
*   **LightGBM**: Algoritmo de Gradient Boosting (Vencedor do AutoML).
*   **Joblib**: Serialização do modelo.

---

## 📂 Estrutura do Projeto

*   `pipeline_precificacao_pycarat.py`: **Pipeline de Treinamento**. Realiza a limpeza dos dados, feature engineering, setup do PyCaret, comparação de modelos e salvamento do melhor modelo (`.pkl`).
*   `resultado.py`: **Script de Inferência**. Carrega o modelo treinado e permite realizar predições para novos imóveis (simulação prática).
*   `Resultado.md`: Relatório detalhado da performance do modelo e defesa técnica.
*   `imoveis_goiania.csv`: Dataset utilizado (Fonte: Web Scraping de portais imobiliários).

---

## 🚀 Como Executar

### 1. Instalar Dependências
```bash
pip install pandas numpy pycaret
```

### 2. Treinar o Modelo
Execute o pipeline para processar os dados e gerar o arquivo `.pkl`:
```bash
python pipeline_precificacao_pycarat.py
```
*Isso criará o arquivo `pipeline_precificacao_pycaret.pkl`.*

### 3. Fazer Predições
Utilize o script de resultado para estimar valores:
```bash
python resultado.py
```
*Edite o arquivo `resultado.py` para alterar os parâmetros do imóvel de teste (bairro, metragem, quartos, etc).*

---

## 🧠 Detalhes do Pipeline

1.  **Limpeza de Dados**: Tratamento de valores nulos, remoção de caracteres de moeda/área.
2.  **Engenharia de Features**: Extração de tipo de imóvel, limpeza de bairros.
3.  **Remoção de Outliers**: Método IQR (Intervalo Interquartil) aplicado a Preço e Metragem.
4.  **Log Transformation**: Aplicação de `log1p` no target (Preço) para normalizar a distribuição e melhorar a performance de modelos lineares e baseados em árvore.
5.  **AutoML**: Setup com validação cruzada (K-Fold) e seleção automática métrica RMSE.

---

## 📝 Autor
Desenvolvido como parte do TCC sobre Precificação de Imóveis com IA.
