# Results of training models
### Parameters
1. Dataset: Juliet Java 1.3
2. Graph construction: compact abstract graph (CAG)
3. Embedding: all-mpnet-base-v2 (embedding length - 768)
4. Number of epochs: 10
5. Learning rate: 0.01


### Results (%)

| GNN Model        |   Accuracy |   Precision |   Recall |    F1 |
|:-----------------|-----------:|------------:|---------:|------:|
| GAT              |      88.85 |       89.53 |    88.85 | 89.02 |
| FeaStNet         |      90.58 |       91.51 |    90.58 | 90.75 |
| UniMP            |      90.76 |       91.66 |    90.76 | 90.93 |
| ARMAConv         |      90.91 |       91.16 |    90.91 | 90.99 |
| GCN              |      91.08 |       91.64 |    91.08 | 91.20 |
| RGGCN            |      91.13 |       92.42 |    91.13 | 91.32 |
