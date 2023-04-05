import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

def main():
    matrixBegin = 0
    geneExpArr = []

    print("Opening File")
    txtFile = open("dupe_fibroblast_data/GSE202991_series_matrix.txt", "r")

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

    plt.title("Gene Expression Dendrogram")
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    #plt.show()
    plt.savefig("dendrogram.png")

if __name__ == "__main__":
	main()
