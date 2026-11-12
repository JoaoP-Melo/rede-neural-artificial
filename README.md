# Rede Neural do Zero para Classificação do Iris

Implementação de uma Rede Neural Artificial com uma camada escondida, construída do zero com **NumPy**, para classificar as três espécies de flores do conjunto de dados **Iris**. O **Scikit-Learn** é usado apenas para carregar e dividir os dados.

## Arquitetura

| Camada   | Neurônios | Ativação |
|----------|-----------|----------|
| Entrada  | 4         | -        |
| Escondida| 8         | ReLU     |
| Saída    | 3         | Softmax  |

## O que o script faz

1. **Pré-processamento**
   - Carrega o Iris com `load_iris`.
   - Divide os dados em 80% treino e 20% teste (estratificado).
   - Padroniza as features (média 0 e desvio padrão 1), usando as estatísticas do treino.
   - Converte os rótulos para One-Hot Encoding.
2. **Inicialização**: pesos (W1, W2) aleatórios com inicialização de He e vieses (b1, b2) zerados.
3. **Treinamento** (1000 épocas, taxa de aprendizado 0.1):
   - Forward pass.
   - Perda Cross-Entropy.
   - Backpropagation com a regra da cadeia.
   - Atualização dos parâmetros por Gradiente Descendente.
4. **Avaliação**: inferência no conjunto de teste e exibição da acurácia final.

## Requisitos

- Python 3.8+
- NumPy
- Scikit-Learn

```bash
pip install numpy scikit-learn
```

## Como executar

```bash
python rede_neural_iris.py
```

## Exemplo de saída

```
Época    1/1000 | Loss: 1.1032 | Acurácia treino: 35.83%
...
Época 1000/1000 | Loss: 0.0489 | Acurácia treino: 98.33%

===== Avaliação Final =====
Loss no teste: 0.0712
Acurácia no teste: 96.67%
```

> Os valores acima são ilustrativos e podem variar conforme a divisão dos dados e a semente aleatória.

## Hiperparâmetros ajustáveis

- `epochs`: número de épocas de treinamento.
- `learning_rate`: taxa de aprendizado.
- `n_hidden`: quantidade de neurônios na camada escondida.
- `random_state` / `np.random.seed`: semente para reprodutibilidade.
