from typing import Dict
import numpy as np


def mahalanobis_distance(
    vec: np.ndarray, mean: np.ndarray, covariance: np.ndarray
) -> float:
    centered_vec = vec - mean
    return centered_vec.dot(np.linalg.solve(covariance, centered_vec)).item()


class GaussianDiscriminantAnalysis:
    def __init__(self, features: Dict[str, np.ndarray]) -> None:
        self.means_: Dict[str, np.ndarray] = {
            class_id: class_features.mean(0)
            for class_id, class_features in features.items()
        }
        self.covariance_ = self._compute_covariance(features)

    def mahalanobis_score(self, feature: np.ndarray) -> float:
        class_conditional_mahalanobis_distances = {
            class_id: mahalanobis_distance(feature, mean, self.covariance_)
            for class_id, mean in self.means_.items()
        }
        distance_minimizing_class = min(
            class_conditional_mahalanobis_distances,
            key=class_conditional_mahalanobis_distances.get,
        )
        return class_conditional_mahalanobis_distances[
            distance_minimizing_class
        ]

    def _compute_covariance(
        self, features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        outer_products = []
        for class_id, class_features in features.items():
            for class_feature in class_features:
                norm_vec = class_feature - self.means_[class_id]
                outer_products.append(np.outer(norm_vec, norm_vec))
        return np.stack(outer_products).mean(0)
