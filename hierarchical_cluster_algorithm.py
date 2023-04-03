import plotly.figure_factory as ff
import numpy as np
import plotext as plt
np.random.seed(1)

def main():
	print("creating dendrogram")
	X = np.random.rand(15, 12) # 15 samples, with 12 dimensions each
	#print(X)
	#fig = ff.create_dendrogram(X)
	#fig.update_layout(width=800, height=500)
	#fig.show()

if __name__ == "__main__":
	main()
