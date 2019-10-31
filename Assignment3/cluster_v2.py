import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sil_score, homogeneity_score, completeness_score
from sklearn.metrics import accuracy_score as acc
from sklearn.mixture import GaussianMixture as EM
import scipy.stats

def cluster_accuracy(Y, labels):
    pred = np.empty_like(Y)
    for label in set(labels):
        mask = labels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return pred

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
        print(sil_score(X, km_labels))
        labeled_clusters = cluster_accuracy(y, km_labels)
        accuracy_scores.append(acc(y, labeled_clusters))
        homo_scores.append(homogeneity_score(y, km_labels))
        completeness_scores.append(completeness_score(y, km_labels))
        ami_scores.append(adjusted_mutual_info_score(y,km_labels))

    plt.plot(cluster_range, sil_scores)
    plt.xlabel('No. Clusters')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score for KMeans: ' + title)
    plt.savefig(folder + '/KMSIL.png')
    plt.close()

    plt.plot(cluster_range, accuracy_scores)
    plt.xlabel('No. Clusters')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Score for KMeans: ' + title)
    plt.savefig(folder + '/KMAccuracy.png')
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
        labeled_clusters = cluster_accuracy(y, em_labels)
        accuracy_scores.append(acc(y, labeled_clusters))
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

    plt.plot(cluster_range, accuracy_scores)
    plt.xlabel('No. Components')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Score for EM: ' + title)
    plt.savefig(folder + '/EMAccuracy.png')
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

def main():
    seed = 200
    df = pd.read_csv("../Dataset/winequality-white.csv", delimiter=";")
    np.random.seed(seed)

    #####load wine data

    lowquality = df.loc[df['quality'] <= 6].index
    highquality = df.loc[df['quality'] > 6].index
    df.iloc[lowquality, df.columns.get_loc('quality')] = 0
    df.iloc[highquality, df.columns.get_loc('quality')] = 1

    X = np.array(df.iloc[:, 0:-1])
    wine_Y = np.array(df.iloc[:, -1])

    standardScalerX = StandardScaler()
    wine_x = standardScalerX.fit_transform(X)


    #####run k means to find best

    kmeans_experiment(wine_x, wine_Y, 'Wine Data', folder="part1_wineplots")

    # Plot Kmeans Wine Cluster

    reduced_data = PCA(n_components=2).fit_transform(wine_x)
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Only for kmeans
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the wine dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('part1_wineplots/kmeans_cluster.png')
    plt.close()

    ###end plot


    ######run em to find best components
    em_experiment(wine_x, wine_Y, 'Wine Data', folder="part1_wineplots")

    # Plot EM wine Cluster

    reduced_data = PCA(n_components=2).fit_transform(wine_x)
    kmeans = EM(n_components=2, n_init=10)
    kmeans.fit(reduced_data)

    h = .02

    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    plt.title('EM clustering on the wine dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('part1_wineplots/em_cluster.png')
    plt.close()


    ####Load digits

    df_digits = pd.read_csv("../Dataset/pendigits.csv", header=None)
    np.random.seed(seed)

    X = np.array(df_digits.iloc[:, 0:-1])
    Y = np.array(df_digits.iloc[:, -1])

    standardScalerX = StandardScaler()
    digits_x = standardScalerX.fit_transform(X)

    # #####run k means to find best
    kmeans_experiment(digits_x, Y, 'Digits Data', folder="part1_digitsplots")

    #Plot Kmeans Digit Cluster

    reduced_data = PCA(n_components=2).fit_transform(digits_x)
    kmeans = KMeans(init='k-means++', n_clusters=9, n_init=10)
    kmeans.fit(reduced_data)

    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('part1_digitsplots/kmeans_cluster.png')
    plt.close()


    ######run em to find best components
    em_experiment(digits_x, Y, 'Digits Data', folder="part1_digitsplots")

    # Plot EM Digit Cluster

    reduced_data = PCA(n_components=2).fit_transform(digits_x)
    kmeans = EM(n_components=8, n_init=10)
    kmeans.fit(reduced_data)
    h = .02
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    plt.title('EM clustering on the digits dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('part1_digitsplots/em_cluster.png')
    plt.close()


if __name__== "__main__":
  main()
