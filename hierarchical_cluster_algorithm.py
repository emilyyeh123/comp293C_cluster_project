'''
Python program to run hiierarchical clustering on dataset

To run program:
python3 hierarchichal_cluster_algorithm.py dataFile.txt dendrograms/dendrogramImageName.png

datasets found on: https://www.ncbi.nlm.nih.gov/ 
--> using Series Matrix txt File(s)

'''

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import mpl_toolkits.mplot3d
from sklearn.cluster import KMeans

def main():
    matrixBegin = 0
    geneExpArr = []

    dataTxtFile = sys.argv[1]
    imageFileName = sys.argv[2]

    print("Opening File")
    #txtFile = open("test_shrunken_dataset/test_set.txt", "r")
    txtFile = open(dataTxtFile, "r")

    for line in txtFile:
        if matrixBegin == 0 and line == "!series_matrix_table_begin\n":
            # check if matrix begin
            matrixBegin = 1
        elif line == "!series_matrix_table_end\n":
            # check if matrix end
            break

        if matrixBegin == 1 and line != "!series_matrix_table_begin\n":
            rowVals = line.split()
            geneExpArr.append(rowVals)

    txtFile.close()

    # create dataframe from input file, store in df
    print("Creating Dataframe")
    df = pd.DataFrame(geneExpArr)
    colHeaders = df.iloc[0].tolist()[1:]
    rowHeaders = df.iloc[:,0].tolist()[1:]
    geneExpArr = geneExpArr[1:]
    geneExpArr = [i[1:] for i in geneExpArr]
    df = df.drop(columns = df.columns[0])
    df = df.drop(df.index[0])
    #print("\n\n", type(df.iloc[0]), "\n", type(df.iloc[:,0]), "\n\n\n")
    #print("\n\n", df.iloc[0].tolist()[1:], "\n", df.iloc[:,0].tolist()[1:], "\n\n\n")
    df = pd.DataFrame(geneExpArr, columns = colHeaders, index = rowHeaders) # FINAL DATAFRAME
    print(df.head())

    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle("Clustering Algorithms on data")#, dataset, "data")

    print("Creating Dendrogram")
    ax1.set_title("Gene Expression Dendrogram")
    dend = shc.dendrogram(shc.linkage(df, method='ward'), ax = ax1)
    #plt.savefig(imageFileName)
    #print("Dendrogram saved to ", imageFileName, "\n")
    #plt.show()

    # Use PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(df.values)
    pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
    print("\n", pca_df)

    # plot data
    print("Performing k-means clustering")
    x = pca_df["PC1"]
    y = pca_df["PC2"]
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pca_df)
    ax2.scatter(x, y, c=kmeans.labels_)
    #ax2.xlabel("PC1")
    #ax2.ylabel("PC2")
    plt.savefig("clustering.png")
    plt.show()
    #print("K-means cluster saved to ", )

if __name__ == "__main__":
	main()
