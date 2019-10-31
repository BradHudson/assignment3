import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import SparseRandomProjection as RCA
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as EM
from sklearn.neural_network import MLPClassifier
import time
import scipy.sparse as sparse
from scipy.linalg import pinv
from matplotlib import offsetbox
from sklearn.metrics import silhouette_score as sil_score, homogeneity_score, completeness_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

def evaluate(clf, training_x, testing_x, training_y, testing_y, title=""):
    start = time.clock()
    clf.fit(training_x, training_y)

    dt_train_time = time.clock() - start
    print('Time to Train: ' + str(dt_train_time))
    print('Training Accuracy: ' + str(clf.score(training_x, training_y)))
    print('Testing Accuracy: ' + str(clf.score(testing_x, testing_y)))
    # print(final_ann.best_params_)
    start = time.clock()
    test_y_predicted = clf.predict(testing_x)
    dt_query_time = time.clock() - start
    print('Time to Query: ' + str(dt_query_time))
    y_true = pd.Series(testing_y)
    y_pred = pd.Series(test_y_predicted)
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    train_sizes, train_scores, test_scores = learning_curve(
        clf,
        training_x,
        training_y, n_jobs=-1,
        cv=5,
        train_sizes=np.linspace(.1, 1.0, 10),
        random_state=200)

    plot_learning_curve(train_scores, test_scores, train_sizes, 'part4plots/' + title + '.png', title= "Learning Curve for " + title)

def plot_learning_curve(train_scores, test_scores, train_sizes, file_name, title=""):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.title(title, fontdict={'size': 16})

    plt.savefig(file_name)
    plt.close()


def PCA_experiment(X, y, title, folder=""):
    pca = PCA(random_state=200).fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(list(range(len(pca.explained_variance_ratio_))), cum_var)
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance: ' + title)
    plt.savefig(folder + '/PCAVariance.png')
    plt.close()

    plt.plot(list(range(len(pca.singular_values_))), pca.singular_values_)
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalues')
    plt.title('PCA Eigenvalues: ' + title)
    plt.savefig(folder + '/PCAEigenvalues.png')
    plt.close()


def ICA_experiment(X, y, title, folder=""):
    n_components_range = list(np.arange(2, X.shape[1], 1))
    ica = ICA(random_state=200)
    kurtosis_scores = []

    for n in n_components_range:
        ica.set_params(n_components=n)
        ice_score = ica.fit_transform(X)
        ice_score = pd.DataFrame(ice_score)
        ice_score = ice_score.kurt(axis=0)
        kurtosis_scores.append(ice_score.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: " + title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(n_components_range, kurtosis_scores)
    plt.savefig(folder + '/ICA.png')
    plt.close()


def RCA_Experiment(X, title, folder=""):
    n_components_range = list(np.arange(2, X.shape[1], 1))
    correlation_coefficient = defaultdict(dict)

    for i, n in product(range(5), n_components_range):
        rp = RCA(random_state=i, n_components=n)
        rp.fit(X)
        projections = rp.components_
        if sparse.issparse(projections):
            projections = projections.todense()
        p = pinv(projections)
        reconstructed = ((p @ projections) @ (X.T)).T
        correlation_coefficient[n][i] = np.nanmean(np.square(X - reconstructed))
    correlation_coefficient = pd.DataFrame(correlation_coefficient).T
    mean_recon = correlation_coefficient.mean(axis=1).tolist()
    std_recon = correlation_coefficient.std(axis=1).tolist()

    plt.plot(n_components_range, mean_recon)
    plt.xlabel('Random Components')
    plt.ylabel('Mean Reconstruction Correlation')
    plt.title('Sparse Random Projection for Mean Reconstruction Correlation: ' + title)
    plt.savefig(folder + '/RcaMeanRE.png')
    plt.close()

    plt.plot(n_components_range, std_recon)
    plt.xlabel('Random Components')
    plt.ylabel('STD Reconstruction Correlation')
    plt.title("Sparse Random Projection for STD Reconstruction Correlation: " + title)
    plt.savefig(folder + '/RcaStdRE.png')
    plt.close()


def run_RFC(X, y, df):
    rfc = RFC(n_estimators=1000, random_state=200, n_jobs=-1)
    important_columns = pd.DataFrame(rfc.fit(X, y).feature_importances_, columns=['Feature Importance'], index=df.columns[0:-1])
    important_columns.sort_values(by=['Feature Importance'], inplace=True, ascending=False)
    important_columns['Cumulative Sum'] = important_columns['Feature Importance'].cumsum()
    important_columns = important_columns[important_columns['Cumulative Sum'] <= 0.95]
    return important_columns, important_columns.index.tolist()

def kmeans_experiment(X, y, title,folder=""):
    cluster_range = list(np.arange(2, 40, 1))
    sil_scores, accuracy_scores, homo_scores, sse_scores, ami_scores = ([] for i in range(5))
    completeness_scores = []

    print(title)
    for k in cluster_range:
        print(k)
        km = KMeans(n_clusters=k).fit(X)
        km_labels = km.predict(X)
        # sse_scores.append(km.score(X))
        sse_scores.append(km.inertia_)
        sil_scores.append(sil_score(X, km_labels))
        # print(sil_score(X, km_labels))
        homo_scores.append(homogeneity_score(y, km_labels))
        completeness_scores.append(completeness_score(y, km_labels))
        ami_scores.append(adjusted_mutual_info_score(y,km_labels))

    plt.plot(cluster_range, sil_scores)
    plt.xlabel('No. Clusters')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score for KMeans: ' + title)
    plt.savefig(folder + '/KMSIL.png')
    plt.close()

    plt.plot(cluster_range, homo_scores)
    plt.xlabel('No. Clusters')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Scores KMeans: ' + title)
    plt.savefig(folder + '/KMHOMOGENEITY.png')
    plt.close()

    plt.plot(cluster_range, sse_scores)
    plt.xlabel('No. Clusters')
    plt.ylabel('SSE Score')
    plt.title('SSE Scores KMeans: ' + title)
    plt.savefig(folder + '/KMSSE.png')
    plt.close()

    plt.plot(cluster_range, ami_scores)
    plt.xlabel('No. Clusters')
    plt.ylabel('AMI Score')
    plt.title('Adjusted Mutual Information Scores KMeans: ' + title)
    plt.savefig(folder + '/KMAMI.png')
    plt.close()

    plt.plot(cluster_range, completeness_scores)
    plt.xlabel('No. Clusters')
    plt.ylabel('Completeness Score')
    plt.title('Completeness Scores KMeans: ' + title)
    plt.savefig(folder + '/KMCompleteness.png')
    plt.close()

def em_experiment(X, y, title,folder=""):
    cluster_range = list(np.arange(2, 11, 1))
    sil_scores, accuracy_scores, homo_scores, sse_scores, ami_scores, bic_scores = ([] for i in range(6))
    completeness_scores = []

    for k in cluster_range:
        # print(k)
        em = EM(n_components=k).fit(X)
        em_labels = em.predict(X)
        sil_scores.append(sil_score(X, em_labels))
        sse_scores.append(em.score(X))
        # print(sil_score(X,em_labels))
        homo_scores.append(homogeneity_score(y, em_labels))
        completeness_scores.append(completeness_score(y, em_labels))
        ami_scores.append(adjusted_mutual_info_score(y,em_labels))
        bic_scores.append(em.bic(X))

    plt.plot(cluster_range, sil_scores)
    plt.xlabel('No. Components')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score for EM: ' + title)
    plt.savefig(folder + '/EMSIL.png')
    plt.close()

    plt.plot(cluster_range, homo_scores)
    plt.xlabel('No. Components')
    plt.ylabel('Homogeneity Score')
    plt.title('Homogeneity Scores EM: ' + title)
    plt.savefig(folder + '/EMHOMOGENEITY.png')
    plt.close()

    plt.plot(cluster_range, completeness_scores)
    plt.xlabel('No. Components')
    plt.ylabel('Completeness Score')
    plt.title('Completeness Score for EM: ' + title)
    plt.savefig(folder + '/EMCompletness.png')
    plt.close()

    plt.plot(cluster_range, sse_scores)
    plt.xlabel('No. Components')
    plt.ylabel('SSE Score')
    plt.title('SSE Scores EM: ' + title)
    plt.savefig(folder + '/EMSSE.png')
    plt.close()

    plt.plot(cluster_range, ami_scores)
    plt.xlabel('No. Components')
    plt.ylabel('AMI Score')
    plt.title('Adjusted Mutual Information Scores EM: ' + title)
    plt.savefig(folder + '/EMAMI.png')
    plt.close()

    plt.plot(cluster_range, bic_scores)
    plt.xlabel('No. Components')
    plt.ylabel('AMI Score')
    plt.title('BIC Scores EM: ' + title)
    plt.savefig(folder + '/EMBIC.png')
    plt.close()


def addclusters(X, km_labels, em_labels):
    df = pd.DataFrame(X)
    df['KM Cluster'] = km_labels
    df['EM Cluster'] = em_labels
    col_1hot = ['KM Cluster', 'EM Cluster']
    df_1hot = df[col_1hot]
    df_1hot = pd.get_dummies(df_1hot).astype('category')
    df_others = df.drop(col_1hot, axis=1)
    df = pd.concat([df_others, df_1hot], axis=1)
    new_X = np.array(df.values, dtype='int64')

    return new_X

### https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_vs_pca.html
def plot_samples_ICA(S, axis_list=None):
    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,
                color='steelblue', alpha=0.5)
    if axis_list is not None:
        colors = ['orange', 'red']
        for color, axis in zip(colors, axis_list):
            axis /= axis.std()
            x_axis, y_axis = axis
            # Trick to get legend to work
            plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)
            plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,
                       color=color)
    plt.hlines(0, -3, 3)
    plt.vlines(0, -3, 3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.xlabel('x')
    plt.ylabel('y')

#### https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            # imagebox = offsetbox.AnnotationBbox(
            #     offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
            #     X[i])
            # ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    return plt

def main():
    df = pd.read_csv("../Dataset/winequality-white.csv", delimiter=";")
    seed = 200
    np.random.seed(seed)

    lowquality = df.loc[df['quality'] <= 6].index
    highquality = df.loc[df['quality'] > 6].index
    df.iloc[lowquality, df.columns.get_loc('quality')] = 0
    df.iloc[highquality, df.columns.get_loc('quality')] = 1

    X = np.array(df.iloc[:, 0:-1])
    wine_Y = np.array(df.iloc[:, -1])

    standardScalerX = StandardScaler()
    wine_x = standardScalerX.fit_transform(X)

    pca_wine = PCA(n_components=7,random_state=seed).fit_transform(wine_x)
    ica_wine = ICA(n_components=9, random_state=seed).fit_transform(wine_x)
    rca_wine = RCA(n_components=8, random_state=seed).fit_transform(wine_x)
    imp_wine, top_columns_wine = run_RFC(wine_x, wine_Y, df)

    rfc_wine = df[top_columns_wine]
    rfc_wine = np.array(rfc_wine.values, dtype='int64')

    X_train, X_test, y_train, y_test = train_test_split(np.array(wine_x), np.array(wine_Y), test_size=0.30)
    learner = MLPClassifier(hidden_layer_sizes=(22,), activation='relu', learning_rate_init=0.0051,
                            random_state=seed)

    evaluate(learner, X_train, X_test, y_train, y_test, title="FullDataset")

    X_train, X_test, y_train, y_test = train_test_split(np.array(pca_wine), np.array(wine_Y), test_size=0.30)
    learner = MLPClassifier(hidden_layer_sizes=(22,), activation='relu', learning_rate_init=0.0051,
                            random_state=seed)

    evaluate(learner, X_train, X_test, y_train, y_test, title="PCA")

    X_train, X_test, y_train, y_test = train_test_split(np.array(ica_wine), np.array(wine_Y), test_size=0.30)
    learner = MLPClassifier(hidden_layer_sizes=(22,), activation='relu', learning_rate_init=0.0051,
                            random_state=seed)

    evaluate(learner, X_train, X_test, y_train, y_test, title="ICA")

    X_train, X_test, y_train, y_test = train_test_split(np.array(rca_wine), np.array(wine_Y), test_size=0.30)
    learner = MLPClassifier(hidden_layer_sizes=(22,), activation='relu', learning_rate_init=0.0051,
                            random_state=seed)

    evaluate(learner, X_train, X_test, y_train, y_test, title="RP")

    X_train, X_test, y_train, y_test = train_test_split(np.array(rfc_wine), np.array(wine_Y), test_size=0.30)
    learner = MLPClassifier(hidden_layer_sizes=(22,), activation='relu', learning_rate_init=0.0051,
                            random_state=seed)

    evaluate(learner, X_train, X_test, y_train, y_test, title="RFC")


if __name__== "__main__":
  main()
