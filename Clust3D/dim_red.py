import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from fancyimpute import SoftImpute


def apply_dim_red(dim_red, data, correlation):
    """
    Apply dimensionality reduction to the input tensor, supporting NaNs via SoftImpute.

    Parameters:
        dim_red (str): Dimensionality reduction method ('none', 'pca_auto', 'pca_elbow', 't-sne', 'ica').
        data (np.ndarray): Input data tensor of shape (N, M, F).
        correlation (list): Correlation list used to restore 3D shape.

    Returns:
        tuple: (reduced_data, data_max, data_min)
            - reduced_data: shape (N, M, d) with d = reduced dimension
            - data_max / data_min: max/min values in the reduced tensor
    """
    data = np.array(data)
    original_shape = data.shape  # (N, M, F)
    flat_data = data.reshape(original_shape[0] * original_shape[1], original_shape[2])  # (N*M, F)

    # --- Handle missing values if needed ---
    if dim_red in ["pca_auto", "pca_elbow", "ica"] and np.isnan(flat_data).any():
        print("Missing values detected → applying SoftImpute before PCA/ICA...")
        flat_data = SoftImpute(max_iters=200, init_fill_method="zero", verbose=False).fit_transform(flat_data)

    # === Apply dimensionality reduction ===
    if dim_red == "pca_elbow":
        pca = PCA()
        pca.fit(flat_data)
        list_pca = np.array(pca.explained_variance_ratio_)

        # Optional elbow plot
        show_plot = False
        if show_plot:
            plt.plot(range(len(list_pca)), list_pca)
            plt.xlabel("Component")
            plt.ylabel("Explained Variance Ratio")
            plt.title("PCA Explained Variance")
            plt.show()

        # Elbow detection
        scaled = MinMaxScaler(feature_range=(0, len(list_pca)))
        a_scaled = scaled.fit_transform(list_pca.reshape(-1, 1)).flatten()
        diffs = np.abs(np.diff(a_scaled))
        pca_index = 2  # default fallback
        for n, value in enumerate(diffs):
            if value <= 1:
                pca_index = n
                break

        print(f"Optimal # of components: {pca_index}")
        print(f"Explained variance: {np.round(100 * np.sum(list_pca[:pca_index]), 2)}%\n")

        pca = PCA(pca_index)
        data_pca = pca.fit_transform(flat_data)

    elif dim_red == "none":
        data_pca = flat_data

    elif dim_red == "pca_auto":
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(flat_data)

    elif dim_red == "t-sne":
        tsne = TSNE()
        data_pca = tsne.fit_transform(flat_data)

    elif dim_red == "ica":
        ica = FastICA()
        data_pca = ica.fit_transform(flat_data)

    else:
        raise ValueError(f"Unknown dimensionality reduction method: {dim_red}")

    # === Reshape to original temporal structure ===
    N = len(correlation)
    M = len(correlation[0]) - 1
    d = data_pca.shape[1]
    data_pca = data_pca.reshape((N, M, d))

    data_min = np.nanmin(data_pca)
    data_max = np.nanmax(data_pca)

    return data_pca, data_max, data_min
