from sklearn.preprocessing import MultiLabelBinarizer

# This custom wrapper is needed to make MultiLabelBinarizer compatible with scikit-learn pipelines
class MLBWrapper:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X)
        return self

    def transform(self, X, y=None):
        return self.mlb.transform(X)

    def get_feature_names_out(self, feature_names):
        return self.mlb.classes_

    def get_params(self, deep=True):
        # Required for scikit-learn compatibility
        return {}

    def set_params(self, **params):
        # Required for scikit-learn compatibility
        return self
