
# 1. Data Preprocessing & Multi-Stage Cleaning

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.mixture import GaussianMixture

# [INPUT]
INPUT_RAW_TRANSIT_DATA = ""
OUTPUT_CLEAN_FEATURES = ""

class DataPreprocessingPipeline:
    def __init__(self, k_neighbors=5, gmm_comp=3, tau=0.05):
        self.k_neighbors = k_neighbors
        self.gmm = GaussianMixture(n_components=gmm_comp, covariance_type='full')
        self.tau = tau

    def parse_raw_data(self, df):
        # 15-min aggregation
        df['depart'] = pd.to_datetime(df['depart'])
        df['depart_15min'] = df['depart'].dt.floor('15T')
        df['stop_route'] = df['stop'].astype(str) + '-' + df['route'].astype(str)

        main_matrix = df.pivot_table(
            index='depart_15min', columns='stop_route',
            values='departure_passengers', aggfunc='sum', fill_value=0
        )
        return main_matrix.values

    def iqr_filter(self, x):
        # Eq (13)-(14)
        q1 = np.percentile(x, 25, axis=0)
        q3 = np.percentile(x, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        missing_mask = (x < lower_bound) | (x > upper_bound)
        x_clipped = np.clip(x, lower_bound, upper_bound)
        return x_clipped, missing_mask

    def knn_imputation(self, x, missing_mask):
        # Eq (15)
        x_imputed = np.copy(x)
        num_time_steps = x.shape[0]

        for t in range(num_time_steps):
            valid_idx = ~missing_mask[t]
            missing_idx = missing_mask[t]

            if missing_idx.any() and valid_idx.any():
                # Fit KNN on valid nodes
                valid_features = np.where(valid_idx)[0].reshape(-1, 1)
                valid_targets = x[t, valid_idx]

                n_neighbors = min(self.k_neighbors, len(valid_features))
                knn = KNeighborsRegressor(n_neighbors=n_neighbors)
                knn.fit(valid_features, valid_targets)

                # Impute missing nodes
                missing_features = np.where(missing_idx)[0].reshape(-1, 1)
                x_imputed[t, missing_idx] = knn.predict(missing_features)

        return x_imputed

    def gmm_filtering(self, x):
        # Eq (16)
        x_flat = x.reshape(-1, 1)
        self.gmm.fit(x_flat)

        prob = np.exp(self.gmm.score_samples(x_flat))
        anomaly_mask = (prob < self.tau).reshape(x.shape)

        #
        node_means = np.mean(x, axis=0)
        x_smoothed = np.where(anomaly_mask, node_means, x)
        return x_smoothed

    def fit_transform(self, raw_df):
        x_raw = self.parse_raw_data(raw_df)
        x_iqr, missing_mask = self.iqr_filter(x_raw)
        x_knn = self.knn_imputation(x_iqr, missing_mask)
        x_clean = self.gmm_filtering(x_knn)
        return x_clean

# Execution Block
if __name__ == "__main__":
    # raw_data = pd.read_csv(INPUT_RAW_TRANSIT_DATA)
    # pipeline = DataPreprocessingPipeline()
    # clean_features = pipeline.fit_transform(raw_data)
    # np.save(OUTPUT_CLEAN_FEATURES, clean_features)
    pass