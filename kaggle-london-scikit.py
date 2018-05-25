import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

x_train = pd.read_csv(
    r'C:\Users\ricardo.soares\.kaggle\competitions\data-science-london-scikit-learn\train.csv')
y_train = pd.read_csv(
    r'C:\Users\ricardo.soares\.kaggle\competitions\data-science-london-scikit-learn\trainLabels.csv')
x_test = pd.read_csv(
    r'C:\Users\ricardo.soares\.kaggle\competitions\data-science-london-scikit-learn\test.csv')

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)

y_train = y_train.ravel()

print('x_train shape: {}\ny_train shape: {}\nx_test shape: {}'.format(
    x_train.shape, y_train.shape, x_test.shape))

x_all = np.r_[x_train, x_test]
print('x_all shape: {}'.format(x_all.shape))

from sklearn.mixture import GaussianMixture
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=cv_type)
        gmm.fit(x_all)
    bic.append(gmm.aic(x_all))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

best_gmm.fit(x_all)
x_train = best_gmm.predict_proba(x_train)
x_test = best_gmm.predict_proba(x_test)

knn = KNeighborsClassifier()
rf = RandomForestClassifier()

param_grid = dict()
grid_search_knn = GridSearchCV(
    knn, param_grid=param_grid, cv=10, scoring='accuracy').fit(x_train, y_train)
print('best estimator knn: {}\nbest score: {}'.format(
    grid_search_knn.best_estimator_, grid_search_knn.best_estimator_.score(x_train, y_train)))
knn_best = grid_search_knn.best_estimator_

grid_search_rf = GridSearchCV(rf, param_grid=dict(
), verbose=3, scoring='accuracy', cv=10).fit(x_train, y_train)
print('best estimator rf: {}\nbest score: {}'.format(
    grid_search_rf.best_estimator_, grid_search_rf.best_estimator_.score(x_train, y_train)))
rf_best = grid_search_rf.best_estimator_

knn_best.fit(x_train, y_train)
print(knn_best.predict(x_test)[0:10])

rf_best.fit(x_train, y_train)
print(rf_best.predict(x_test)[0:10])

print('score for knn: {}'.format(cross_val_score(
    knn_best, x_train, y_train, cv=10, scoring='accuracy').mean()))
print('score for rf: {}'.format(cross_val_score(
    rf_best, x_train, y_train, cv=10, scoring='accuracy').max()))

knn_best_pred = pd.DataFrame(knn_best.predict(x_test))
rf_best_pred = pd.DataFrame(rf_best.predict(x_test))

knn_best_pred.index += 1
rf_best_pred.index += 1

knn_best_pred.to_csv(
    r'C:\Users\ricardo.soares\.kaggle\competitions\data-science-london-scikit-learn\knn_best_pred.csv')
rf_best_pred.to_csv(
    r'C:\Users\ricardo.soares\.kaggle\competitions\data-science-london-scikit-learn\rf_best_pred.csv')
