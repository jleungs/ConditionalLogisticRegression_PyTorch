import threading
import itertools
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score


class GridSearchKFoldCV():
    def __init__(self, model, param_grid, n_threads=4, k=5):
        self.model = model
        self.param_grid = param_grid
        self.k = k
        self.n_threads = n_threads
        self.score_dict = dict()
        # cartesian product of the hyperparameters passed
        param_list = list(itertools.product(*param_grid.values()))
        # split params for threads
        self.param_list_split = np.array_split(param_list, n_threads)


    def fit(self, X, y, strata):
        self.X = X
        self.y = y
        self.strata = strata

        splitter = GroupShuffleSplit(n_splits=self.k, test_size=0.2)
        self.train_idx_list = list()
        self.test_idx_list = list()
        # Perform the split
        for train_idx, test_idx in splitter.split(X, groups=strata):
            # save index to a list
            self.train_idx_list.append(train_idx)
            self.test_idx_list.append(test_idx)
        # loop for each thread
        thread_list = list()
        for t in range(self.n_threads):
            x = threading.Thread(target=self.thread_main, args=(self.param_list_split[t],))
            thread_list.append(x)
            x.start()
        # wait for all threads to finish
        for t in thread_list:
            t.join()


    def thread_main(self, param_list):
        for param in param_list:
            param = param.tolist()
            param_dict = dict(zip(self.param_grid.keys(), param))
            model = self.model(**param_dict, verbose=False)
            fold_score = list()
            for i in range(self.k):
                train_idx = self.train_idx_list[i]
                test_idx = self.test_idx_list[i]
                model.fit(self.X[train_idx], self.y[train_idx], self.strata[train_idx])
                score = self.evaluate_model(model, test_idx)
                fold_score.append(score)
            with threading.Lock():
                self.score_dict[str(param_dict)] = np.mean(fold_score)


    def evaluate_model(self, model, idx):
        # sensitivity/recall for true positive correct
        y_pred = model.predict(self.X[idx], self.strata[idx])
        return recall_score(self.y[idx], y_pred)


    def best_score(self):
        return sorted(self.score_dict.items(), key=lambda item: item[1], reverse=True)[0]

