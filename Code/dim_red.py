import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE



def apply_dim_red(dim_red, data, correlation):

    # Automatic selection of the number of principal components based on the elbow rule
    if dim_red == "pca_elbow":

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

        pca = PCA()
        pca.fit_transform(data)
        list_pca = np.array(pca.explained_variance_ratio_)


        show_plot = False
        if show_plot:

            # PCA explained variance plot
            plt.plot(range(len(list_pca)), list_pca)
            plt.show()


        scaler = MinMaxScaler(feature_range=(0, len(list_pca)))
        a_scaled = scaler.fit_transform(np.array(list_pca).reshape(-1, 1))
        a_scaled = np.array(a_scaled).reshape(len(a_scaled), )

        c = np.diff(a_scaled)
        c = abs(np.array(c))

        for n, value in enumerate(c):
            if value <= 1:
                pca_index = n
                break

        print(f"Optimal no. of principal components: {pca_index}")
        print(f"Explained variance with {pca_index} principal components: {np.round(100 * np.sum(list_pca[:pca_index]), 2)}%")
        print()
        pca = PCA(pca_index)
        data_pca = pca.fit_transform(data)

        data_pca = np.array(data_pca).reshape((len(correlation), len(correlation[0]) - 1, data_pca.shape[1]))


        kk_max = []
        kk_min = []
        for kk in data_pca:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        input_data = data_pca


    # No pca
    elif dim_red == "none":

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

        kk_max = []
        kk_min = []
        for kk in data:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        data = np.array(data).reshape((len(correlation), len(correlation[0]) - 1, data.shape[1]))

        input_data = data



    # PCA using default PCA parameters (All components are kept)
    elif dim_red == "pca_auto":

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

        pca = PCA(2)
        data_pca = pca.fit_transform(data)
        data_pca = np.array(data_pca).reshape((len(correlation), len(correlation[0]) - 1, data_pca.shape[1]))


        kk_max = []
        kk_min = []
        for kk in data_pca:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        input_data = data_pca





    # t-SNE
    elif dim_red == "t-sne":

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

        tsne = TSNE()
        data_pca = tsne.fit_transform(data)

        kk_max = []
        kk_min = []
        for kk in data_pca:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        data_pca = np.array(data_pca).reshape((len(correlation), len(correlation[0]) - 1, data_pca.shape[1]))

        input_data = data_pca


    # UMAP
    elif dim_red == "umap":
        import umap

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])
        umap_model = umap.UMAP()

        # Fit the UMAP model to the data
        umap_results = umap_model.fit_transform(data)


        kk_max = []
        kk_min = []
        for kk in umap_results:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        data_pca = np.array(umap_results).reshape((len(correlation), len(correlation[0]) - 1, umap_results.shape[1]))

        input_data = data_pca




    # ICA
    elif dim_red == "ica":
        from sklearn.decomposition import FastICA

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])
        ica = FastICA()

        # Fit the ICA model to the mixed signal
        ica.fit(data)

        # Transform the mixed signal into its independent components
        data_pca = ica.transform(data)

        kk_max = []
        kk_min = []
        for kk in data_pca:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        data_pca = np.array(data_pca).reshape((len(correlation), len(correlation[0]) - 1, data_pca.shape[1]))

        input_data = data_pca




    return input_data, data_max, data_min






