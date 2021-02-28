import pickle
import sys

from sklearn.linear_model import LogisticRegression
from features_extraction import build_df, get_y

train_df, V, vocab = build_df(file=sys.argv[1], V=None, vocab=sys.argv[3])
train_y = get_y(file=sys.argv[2], df=train_df)
print(f'Train size: {train_df.shape}, y: {train_y.shape}, y=1: {train_y[train_y == 1].shape}')

lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.01).fit(train_df, train_y)

with open('trained_model', 'wb') as f:
    pickle.dump((lr_clf, V, vocab), f, pickle.HIGHEST_PROTOCOL)
