# Cassava Leaf Disease Classification 🌿

Repository for a notebook from a Deep Learning class (IBMEC).

Pipeline completo de visão computacional que classifica imagens de folhas de mandioca (*cassava*) em **5 categorias** (4 doenças + folha saudável), utilizando *transfer learning* com **ResNet50** pré-treinada na ImageNet. Projeto baseado no dataset da competição Kaggle [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification).

## Grupo

- Thiago Brandão
- Fabiano Amorim
- Breno França
- João Pedro Menezes
- Isabella Cristina

## Arquivo

### `AP2_projeto_deep_learning.ipynb`

Notebook único, organizado em três grandes etapas:

**1. EDA (Análise Exploratória)**
- Download do dataset via `kagglehub` (21.397 imagens + `train.csv` + mapa de classes).
- Análise da distribuição das classes — dataset fortemente desbalanceado: *Cassava Mosaic Disease* (CMD) representa ~61% das imagens, enquanto *Cassava Bacterial Blight* (CBB) é só ~5%.
- Inspeção visual de amostras de cada classe e análise das dimensões das imagens (todas 800x600).

**2. Pré-processamento**
- Split estratificado 70/10/20 (treino/validação/teste), preservando a proporção de classes.
- Pipeline `tf.data` com resize para 336x336, normalização no padrão ResNet e *data augmentation* (flip, rotação, recorte) aplicado apenas no treino — dobrando o volume de treino (~15k → ~30k imagens).
- Shuffle, batching (128) e prefetch para otimizar o carregamento.

**3. Modelo**
- **Arquitetura**: ResNet50 pré-treinada (ImageNet) como backbone + cabeça densa customizada (Dense 512 → 256 → 32, com BatchNorm, Dropout e regularização L1L2).
- **Class weights** balanceados para compensar o desbalanceamento das classes.
- **Métrica customizada** (`MeanRecall`): recall médio por classe, mais informativo que acurácia em dataset desbalanceado.
- **Treino em 2 fases**:
  - *Fase 1 — Feature extraction*: backbone todo congelado, só a cabeça treina. `val_mean_recall` sobe de ~0.51 para ~0.66.
  - *Fase 2 — Fine-tuning*: descongela o bloco `conv5_block` da ResNet. `val_mean_recall` chega a ~0.70 (com leve overfitting entre treino e validação).
- **Avaliação final no teste**: ~70% de mean recall e ~82% de acurácia geral, com desempenho forte na classe majoritária (CMD, F1 ~0.93) e mais modesto nas classes raras — coerente com o desafio imposto pelo desbalanceamento.

## Dataset

- **Fonte**: [Kaggle — Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification)
- **Tamanho**: 21.397 imagens, 5 classes
- **Classes**: Cassava Bacterial Blight (CBB), Cassava Brown Streak Disease (CBSD), Cassava Green Mottle (CGM), Cassava Mosaic Disease (CMD), Healthy

## Como rodar

Desenvolvido no Google Colab. Requer uma `kaggle.json` (API key do Kaggle) para baixar o dataset da competição. Principais dependências:

```bash
pip install tensorflow kagglehub pandas numpy matplotlib scikit-learn pillow
```

## Resultados e comparação com referências

| Etapa | Mean Recall (val) |
|---|---|
| Feature extraction (Fase 1) | ~0.66 |
| Fine-tuning conv5 (Fase 2) | ~0.70 |
| **Teste final** | **~0.70 recall / ~82% acurácia** |

Para efeito de comparação: a referência [1] (também com ResNet50, mas descongelando o modelo inteiro no fine-tuning) obteve 86% de acurácia, e a referência [3] indica que os melhores modelos da competição (muitos ensembles) chegam a 89-91%. Isso ajuda a dimensionar o quão desafiador é o problema mesmo com uma arquitetura robusta.

## Referências

1. https://lazylovable.medium.com/cassava-leaf-disease-classification-with-deep-learning-part-iii-3ca80722a209
2. Géron, Aurélien. *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media, 2022.
3. https://chat.deepseek.com/share/csph3ui7lry015oviv
