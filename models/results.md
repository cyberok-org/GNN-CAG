## Model training: FeaStNet, ARMAConv, GCN
The results of predicting the presence of vulnerabilities in the code on the Juliet dataset using CAGs:

|          | Accuracy | Precision | Recall |   F1 |
|:---------|---------:|----------:|-------:|-----:|
| FeaStNet |     88.3 |      74.0 |   96.0 | 84.0 |
| ARMAConv |     91.0 |      81.8 |   93.0 | 87.1 |
| GCN      |     91.2 |      80.0 |   96.0 | 87.0 |


| ![FeaStNet](/images/FeaStNet_10epochs.png) |
|:------------------------------------------:|
|                 *FeaStNet*                 |


| ![ARMAConv](/images/ARMA_10epochs.png) |
|:--------------------------------------:|
|               *ARMAConv*               |


| ![GCN](/images/GCN_10epochs.png) |
|:--------------------------------:|
|              *GCN*               |

