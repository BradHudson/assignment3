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
from sklearn.metrics import silhouette_score as sil_score, homogeneity_score
from sklearn.mixture import GaussianMixture as EM
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import timeit
from sklearn.metrics import accuracy_score
import scipy.sparse as sparse
from scipy.linalg import pinv
from matplotlib import offsetbox
from sklearn.metrics import silhouette_score as sil_score, homogeneity_score, completeness_score

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

    #### Question 2

    ### Examine first two components of PCA

    pca = PCA(n_components=2)
    pca.fit(wine_x)

    X_pca = pca.transform(wine_x)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)

    X_new = pca.inverse_transform(X_pca)
    plt.scatter(wine_x[:, 0], wine_x[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.savefig('part2_wineplots' + '/PCA_TwoD.png')
    plt.close()

    ### Examine first two components of ICA

    ica = ICA(n_components=2)
    ica.fit(wine_x)

    X_ica = ica.transform(wine_x)
    X_ica /= X_ica.std(axis=0)

    plt.figure()
    plt.subplot(2, 2, 1)
    plot_samples_ICA(wine_x[:,[0,1]] / wine_x[:,[0,1]].std())
    plt.title('True Independent Sources')

    plt.subplot(2, 2, 4)
    plot_samples_ICA(X_ica / np.std(X_ica))
    plt.title('ICA recovered signals')

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
    plt.savefig('part2_wineplots/ICA_plot.png')
    plt.close()

    ### Examine first two components of Sparse RP

    rp = RCA(n_components=2, random_state=42)
    X_projected = rp.fit_transform(wine_x)
    rp_plot = plot_embedding(X_projected, wine_Y, "Random Projection of the Wine")
    rp_plot.savefig('part2_wineplots/RCA_plot.png')
    plt.close()


    #### Experiments DR

    PCA_experiment(wine_x, wine_Y, "Wine Data", folder="part2_wineplots")
    ICA_experiment(wine_x, wine_Y, "Wine Data", folder="part2_wineplots")
    RCA_Experiment(wine_x, "Wine Data", folder="part2_wineplots")
    imp_wine, top_columns = run_RFC(wine_x, wine_Y, df)
    print(imp_wine)
    print('***********')

    pca_wine = PCA(n_components=7,random_state=seed).fit_transform(wine_x)
    ica_wine = ICA(n_components=8, random_state=seed).fit_transform(wine_x)
    rca_wine = RCA(n_components=5, random_state=seed).fit_transform(wine_x)

    rfc_wine = df[top_columns]
    rfc_wine = np.array(rfc_wine.values, dtype='int64')

    #### End Question 2

    ### Question 3

    pca_wine = PCA(n_components=7,random_state=seed).fit_transform(wine_x)
    ica_wine = ICA(n_components=9, random_state=seed).fit_transform(wine_x)
    rca_wine = RCA(n_components=8, random_state=seed).fit_transform(wine_x)
    imp_wine, top_columns_wine = run_RFC(wine_x, wine_Y, df)

    rfc_wine = df[top_columns_wine]
    rfc_wine = np.array(rfc_wine.values, dtype='int64')

    kmeans_experiment(pca_wine, wine_Y, "PCA Wine Data", folder="part3_wineplots/PCA")
    kmeans_experiment(ica_wine, wine_Y, "ICA Wine Data", folder="part3_wineplots/ICA")
    kmeans_experiment(rca_wine, wine_Y, "Random Projection Wine Data", folder="part3_wineplots/RP")
    kmeans_experiment(rfc_wine, wine_Y, "RFC Wine Data", folder="part3_wineplots/RFC")

    em_experiment(pca_wine, wine_Y, "PCA Wine Data", folder="part3_wineplots/PCA")
    em_experiment(ica_wine, wine_Y, "ICA Wine Data", folder="part3_wineplots/ICA")
    em_experiment(rca_wine, wine_Y, "Random Projection Wine Data", folder="part3_wineplots/RP")
    em_experiment(rfc_wine, wine_Y, "RFC Wine Data", folder="part3_wineplots/RFC")

    ### End Question 3

    #### Digits data

    df_digits = pd.read_csv("../Dataset/pendigits.csv", header=None)
    np.random.seed(seed)

    X = np.array(df_digits.iloc[:, 0:-1])
    digits_Y = np.array(df_digits.iloc[:, -1])

    standardScalerX = StandardScaler()
    digits_x = standardScalerX.fit_transform(X)


    #### Plot PCA

    pca = PCA(n_components=2)
    pca.fit(digits_x)

    X_pca = pca.transform(digits_x)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)

    X_new = pca.inverse_transform(X_pca)
    plt.scatter(digits_x[:, 0], digits_x[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.savefig('part2_digitsplots' + '/PCA_TwoD.png')
    plt.close()

    ### Examine first two components of ICA

    ica = ICA(n_components=2)
    ica.fit(digits_x)

    X_ica = ica.transform(digits_x)
    X_ica /= X_ica.std(axis=0)

    plt.figure()
    plt.subplot(2, 2, 1)
    plot_samples_ICA(digits_x[:, [0, 1]] / digits_x[:, [0, 1]].std())
    plt.title('True Independent Sources')

    plt.subplot(2, 2, 4)
    plot_samples_ICA(X_ica / np.std(X_ica))
    plt.title('ICA recovered signals')

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)
    plt.savefig('part2_digitsplots/ICA_plot.png')
    plt.close()

    ### Examine first two components of Sparse RP

    rp = RCA(n_components=2, random_state=42)
    X_projected = rp.fit_transform(digits_x)
    rp_plot = plot_embedding(X_projected, digits_Y, "Random Projection of the Digits")
    rp_plot.savefig('part2_digitsplots/RCA_plot.png')
    plt.close()

    ### Question 2

    PCA_experiment(digits_x, digits_Y, "Digits Data", folder="part2_digitsplots")
    ICA_experiment(digits_x, digits_Y, "Digits Data", folder="part2_digitsplots")
    RCA_Experiment(digits_x, "Digits Data", folder="part2_digitsplots")
    imp_digits, top_columns = run_RFC(digits_x, digits_Y, df_digits)
    print('***********')
    print(imp_digits)

    pca_digits = PCA(n_components=8, random_state=seed).fit_transform(digits_x)
    ica_digits = ICA(n_components=11, random_state=seed).fit_transform(digits_x)
    rca_digits = RCA(n_components=8, random_state=seed).fit_transform(digits_x)

    rfc_digits = df_digits[topcols_digits]
    rfc_digits = np.array(rfc_digits.values, dtype='int64')

    kmeans_experiment(pca_digits, digits_Y, "PCA Digits Data", folder="part3_digitsplots/PCA")
    kmeans_experiment(ica_digits, digits_Y, "ICA Digits Data", folder="part3_digitsplots/ICA")
    kmeans_experiment(rca_digits, digits_Y, "RCA Digits Data", folder="part3_digitsplots/RP")
    kmeans_experiment(rfc_digits, digits_Y, "RFC Digits Data", folder="part3_digitsplots/RFC")

    em_experiment(pca_digits, digits_Y, "PCA Digits Data", folder="part3_digitsplots/PCA")
    em_experiment(ica_digits, digits_Y, "ICA Digits Data", folder="part3_digitsplots/ICA")
    em_experiment(rca_digits, digits_Y, "RCA Digits Data", folder="part3_digitsplots/RP")
    em_experiment(rfc_digits, digits_Y, "RFC Digits Data", folder="part3_digitsplots/RFC")

    ### End Question 2

    ### Question 3

    pca_digits = PCA(n_components=7,random_state=seed).fit_transform(digits_x)
    ica_digits = ICA(n_components=9, random_state=seed).fit_transform(digits_x)
    rca_digits = RCA(n_components=8, random_state=seed).fit_transform(digits_x)
    imp_digits, top_columns_digits = run_RFC(digits_x, digits_Y, df_digits)

    rfc_digits = df_digits.drop(df_digits.columns[12], axis=1)
    rfc_digits = rfc_digits.drop(rfc_digits.columns[2], axis=1)
    rfc_digits = np.array(rfc_digits.values, dtype='int64')

    kmeans_experiment(pca_digits, digits_Y, "PCA Digits Data", folder="part3_digitsplots/PCA")
    kmeans_experiment(ica_digits, digits_Y, "ICA Digits Data", folder="part3_digitsplots/ICA")
    kmeans_experiment(rca_digits, digits_Y, "Random Projection Digits Data", folder="part3_digitsplots/RP")
    kmeans_experiment(rfc_digits, digits_Y, "RFC Digits Data", folder="part3_digitsplots/RFC")

    em_experiment(pca_digits, digits_Y, "PCA Digits Data", folder="part3_digitsplots/PCA")
    em_experiment(ica_digits, digits_Y, "ICA Digits Data", folder="part3_digitsplots/ICA")
    em_experiment(rca_digits, digits_Y, "Random Projection Digits Data", folder="part3_digitsplots/RP")
    em_experiment(rfc_digits, digits_Y, "RFC Digits Data", folder="part3_digitsplots/RFC")

    ### End Question 3


if __name__== "__main__":
  main()
