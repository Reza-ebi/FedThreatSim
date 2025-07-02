import numpy as np

class ServerNode:
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.round = 0

    def aggregate_models(self, model_updates):
        avg = np.mean(model_updates, axis=0)
        stds = np.std(model_updates, axis=0)
        filtered = []

        for update in model_updates:
            if np.all(np.abs(update - avg) < self.threshold * stds):
                filtered.append(update)
            else:
                print(f"[!] Outlier detected. Model excluded.")

        if filtered:
            final_model = np.mean(filtered, axis=0)
        else:
            final_model = avg
        return final_model
