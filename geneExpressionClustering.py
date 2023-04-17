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
    datasetName = sys.argv[2] # only use "_" as spaces

    print("Opening .txt File")
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
    print("\nCreating Dataframe")
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

    print("\nPerforming Hierarchical Clustering")
    figDendrogram = plt.figure()
    plt.title("Hierarchical Clustering On " + datasetName.replace("_", " ") + " Data")
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    plt.savefig("figures/"+ datasetName +"_dendrogram.png")
    print("Dendrogram saved to figures/"+ datasetName +"_dendrogram.png\n")
    #plt.show()

    # Use PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(df.values)
    pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])

    # plot data
    print("\nPerforming k-means clustering")
    figKMeans = plt.figure()
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pca_df)
    plt.scatter(pca_df["PC1"], pca_df["PC2"], c=kmeans.labels_)
    plt.title("K-Means Clustering On " + datasetName.replace("_", " ") + " Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig("figures/"+ datasetName +"_KMeans.png")
    print("K-means plot saved to figures/"+ datasetName +"_KMeans.png")
    #plt.show()

if __name__ == "__main__":
	main()
