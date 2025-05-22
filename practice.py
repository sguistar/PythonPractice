import pandas as pd
import numpy as np
import warp as wp
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv('train.csv', names=None)

# 删除重复项前显示重复行
print(data[data.duplicated(keep=False)])
print("原始数据形状:", data.shape)

# 删除 id 和重复项
data = data.drop(['id'], axis=1)
data.drop_duplicates(inplace=True)

# 处理缺失值
data['default'] = data['default'].replace('unknown', wp.nan)
data['default'] = data['default'].fillna(data['default'].mode()[0])

# LabelEncode job
le_job = preprocessing.LabelEncoder()
data['job'] = le_job.fit_transform(data['job'])

# LabelEncode 多列
cat_cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact',
            'month', 'day_of_week', 'poutcome']

for col in cat_cols:
    le = preprocessing.LabelEncoder()
    data[col] = le.fit_transform(data[col])

# 目标变量 y
y = data['subscribe']
le_subscribe = preprocessing.LabelEncoder()
y = le_subscribe.fit_transform(y)  # 转换为 0/1

# 特征变量 x
x = data.drop(['subscribe'], axis=1)

# PCA降维
pca = PCA()
x_pca = pca.fit_transform(x)

# 显示解释方差比例
print("各主成分方差比例:", pca.explained_variance_ratio_)
print("累计方差比例:", np.cumsum(pca.explained_variance_ratio_))

# 特征缩放
scaler = MinMaxScaler((0.01, 0.99))
x_scaled = scaler.fit_transform(x_pca)
x_scaled = pd.DataFrame(x_scaled)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# 支持向量机模型
clf = SVC(random_state=0, kernel='rbf', gamma=0.3, C=13)
clf.fit(X_train, y_train)

# 模型评分
print("模型准确率:", clf.score(X_test, y_test))

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

models = {
    'logreg': {
        'model': LogisticRegression(max_iter=500),
        'params': {
            'C': [0.01, 0.1, 1, 10]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3,5,7,9]
        }
    },
    'rf': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 300],
            'max_depth': [None, 5, 10]
        }
    },
    'gb': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }
    },
    'ada': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0]
        }
    },
    'xgb': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5]
        }
    },
    'mlp': {
        'model': MLPClassifier(max_iter=1000, random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [1e-4, 1e-3]
        }
    }
}

best_models = {}
for name, mp in models.items():
    print(f"\n🔍 Tuning {name}…")
    grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f" → Best {name}: {grid.best_score_:.4f} with {grid.best_params_}")
    best_models[name] = grid.best_estimator_


for name, mdl in best_models.items():
    acc = mdl.score(X_test, y_test)
    print(f"{name.upper():8} Test accuracy: {acc:.4f}")

voting_clf = VotingClassifier(
    estimators=[(n, best_models[n]) for n in ['rf','gb','logreg']],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
print("VOTING   Test accuracy:", voting_clf.score(X_test, y_test))
