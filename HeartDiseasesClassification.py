# PROJEKT EKSPLORACJI DANYCH AUTORSTWA PATRYKA PŁÓCIENNIKA, 2022r.

# ZAŁĄCZONE BIBLIOTEKI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes

# POBIERANIE DANYCH Z PLIKU
df = pd.read_csv('Cleveland_heart_disease_database\clevelandDataBinary.csv')

# ANALIZA STRUKTURY BAZY
print(df.columns)
print(df.describe())
print(df.info())
print(df.sample(5))

# LICZBA ZDROWYCH (ETYKIETA 0) LUB CHORYCH
# (ETYKIETA 1,2,3,4 W ZALEŻNOŚCI OD ZAAWANSOWANIA CHOROBY / ETYKIETA 1 GDY CHORY) PACJENTÓW
print(df[['num']].value_counts())

# PRZYKŁADOWY WYKRES VIOLINPLOT
sns.violinplot(data=df, y='num', x='sex')

# PRZYKŁADOWY WYKRES PAIRPLOT
sns.pairplot(data=df.drop(columns=['age', 'trestbps', 'chol', 'fbs', 'thalach', 'oldpeak', 'slope']), hue='num')
plt.show()

# WSZYSTKIE ETYKIETY PRZED EKSTRAKCJĄ ['age', 'sex', 'cp', 'trestbps', 'restecg', 'chol', 'fbs', 'thalach', 'exang',
# 'oldpeak', 'slope', 'ca', 'thal']

# WYBÓR CECH I ETYKIET (PO 1 URUCHOMIENIU WYBRANE ZOSTAŁO 8 z 13 NAJISTOTNIEJSZYCH CECH)
xnames = ['age', 'sex', 'cp', 'trestbps', 'restecg', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
          'thal']
yname = 'num'

X = df[xnames].to_numpy()
Y = df[yname].to_numpy()

# PODZIAŁ NA ZBIÓR DO WYBORU MODELU I ZBIÓR DO KLASYFIKACJI
X_model, X_class, Y_model, Y_class = train_test_split(X, Y, test_size=0.45, random_state=22)

# WYBÓR MODELU DO KLASYFIKACJI - WYKORZYSTANIE 55% DANYCH ZBIORU

# WALIDACJA KRZYŻOWA KFOLD
kf = KFold(n_splits=5)
kf.get_n_splits(X_model)

# RANDOM FOREST (RANDOMOWY LAS)
scoreRF = cross_val_score(ensemble.RandomForestClassifier(random_state=22),
                          X_model, Y_model, cv=kf, scoring="accuracy")
print(f'Wynik każdego z foldów RF: {scoreRF}')
print(f'Uśredniony wynik Random Forest: {"{:.2f}".format(scoreRF.mean())}')

# LOGISTIC REGRESSION (REGRESJA LOGISTYCZNA)
scoreLR = cross_val_score(linear_model.LogisticRegression(random_state=22, max_iter=20000),
                          X_model, Y_model, cv=kf, scoring="accuracy")
print(f'Wynik każdego z foldów LR: {scoreLR}')
print(f'Uśredniony wynik Logistic Regression: {"{:.2f}".format(scoreLR.mean())}')

# DECISION TREE (DRZEWO DECYZYJNE)
scoreDT = cross_val_score(tree.DecisionTreeClassifier(random_state=22),
                          X_model, Y_model, cv=kf, scoring="accuracy")
print(f'Wynik każdego z foldów DT: {scoreDT}')
print(f'Uśredniony wynik Decision Tree: {"{:.2f}".format(scoreDT.mean())}')

# SUPPORT VECTOR CLASSIFIER (KLASYFIKATOR WEKTORÓW POMOCNICZYCH)
scoreSVC = cross_val_score(svm.SVC(kernel='linear', random_state=22),
                           X_model, Y_model, cv=kf, scoring="accuracy")
print(f'Wynik każdego z foldów SVC: {scoreSVC}')
print(f'Uśredniony wynik Support Vector Classifier : {"{:.2f}".format(scoreSVC.mean())}')

# GAUSSIAN NAIVE BAYES (NAIWNY KLASYFIKATOR BAYESOWSKI)
scoreGNB = cross_val_score(naive_bayes.GaussianNB(),
                           X_model, Y_model, cv=kf, scoring="accuracy")
print(f'Wynik każdego z foldów GNB: {scoreGNB}')
print(f'Uśredniony wynik Gaussian Naive Bayes : {"{:.2f}".format(scoreGNB.mean())}')

# KLASYFIKACJA W OPARCIU O WYBRANY MODEL - WYKORZYSTANIE POZOSTAŁYCH 45% DANYCH ZBIORU

# PODZIAŁ NA ZBIÓR UCZĄCY I TRENINGOWY
X_train, X_test, Y_train, Y_test = train_test_split(X_class, Y_class, test_size=0.3, random_state=22)

# ZBUDOWANIE MODELU REGRESJI LOGISTYCZNEJ
model = linear_model.LogisticRegression(random_state=10, max_iter=20000)

# NAUKA MODELU ZA POMOCĄ DANYCH TRENINGOWYCH
model.fit(X_train, Y_train)

# KLASYFIKACJA ZA POMOCĄ DANYCH TESTOWYCH
Y_pred = model.predict(X_test)

# WYZNACZENIE POZIOMU ISTOTNOŚCI CECH DLA MODELU REGRESJI LOGISTYCZNEJ
importance = model.coef_[0]
for i, v in enumerate(importance):
    print('Cecha: %0d, Istotność: %.5f' % (i, v))

# WYZNACZENIE POPRAWNOŚCI KLASYFIKACJI (ACCURACY) DLA REGRESJI LOGISTYCZNEJ
print('Poprawność klasyfikacji za pomocą modelu regresji logistycznej na zbiorze testowym: {:.2f}'.format(
    model.score(X_test, Y_test)))

# MACIERZ POMYŁEK (CONFUSION MATRIX)
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)

# RAPORT Z PROCESU KLASYFIKACJI
print(classification_report(Y_test, Y_pred))

# KRZYWA ROC
logit_roc_auc = roc_auc_score(Y_test, Y_pred)
fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Regresja Logistyczna (obszar = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Odsetek uznany jako fałszywie pozytywny')
plt.ylabel('Odsetek uznany jako prawdziwie pozytywny')
plt.title('ROC - Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()