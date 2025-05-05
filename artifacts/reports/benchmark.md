### Test set

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Legacy v1 (TF-IDF + LinearSVC) | 0.833 | 0.907 | 0.792 | 0.845 | 0.832 | - |
| TF-IDF + LogReg | 0.896 | 0.904 | 0.916 | 0.910 | 0.893 | 0.959 |
| BiLSTM + Attention | 0.895 | 0.906 | 0.911 | 0.909 | 0.892 | 0.952 |
| DistilRoBERTa | 0.926 | 0.937 | 0.935 | 0.936 | 0.924 | 0.972 |

### Test (original data)

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Legacy v1 (TF-IDF + LinearSVC) | 0.966 | 0.964 | 0.980 | 0.972 | 0.964 | - |
| TF-IDF + LogReg | 0.924 | 0.922 | 0.956 | 0.939 | 0.920 | 0.976 |
| BiLSTM + Attention | 0.930 | 0.929 | 0.958 | 0.943 | 0.926 | 0.970 |
| DistilRoBERTa | 0.940 | 0.944 | 0.958 | 0.951 | 0.937 | 0.980 |

### Obfuscated test set

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Legacy v1 (TF-IDF + LinearSVC) | 0.708 | 0.893 | 0.560 | 0.688 | 0.707 | - |
| TF-IDF + LogReg | 0.889 | 0.910 | 0.896 | 0.903 | 0.887 | 0.954 |
| BiLSTM + Attention | 0.882 | 0.911 | 0.882 | 0.896 | 0.880 | 0.944 |
| DistilRoBERTa | 0.914 | 0.941 | 0.907 | 0.924 | 0.912 | 0.968 |

### Challenge set

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Legacy v1 (TF-IDF + LinearSVC) | 0.628 | 0.757 | 0.583 | 0.659 | 0.625 | - |
| TF-IDF + LogReg | 0.897 | 0.885 | 0.958 | 0.920 | 0.889 | 0.949 |
| BiLSTM + Attention | 0.910 | 0.918 | 0.938 | 0.928 | 0.905 | 0.960 |
| DistilRoBERTa | 0.910 | 0.977 | 0.875 | 0.923 | 0.908 | 0.973 |
