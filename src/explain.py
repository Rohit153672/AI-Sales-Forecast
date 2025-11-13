import shap

class SHAPExplain:
    def __init__(self, model):
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X):
        return self.explainer.shap_values(X)


