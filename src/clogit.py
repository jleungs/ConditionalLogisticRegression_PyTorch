import torch
import random
import pandas as pd
import numpy as np


class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ConditionalLogisticRegression(torch.nn.Module):
    """
    Conditional / Fixed-Effects Logistic Regression model implemented with PyTorch.
    """
    def __init__(self, lr=0.00001, epochs=100, groups_batch_size=1, l2_constant=0.0001, verbose=True, earlystop_patience=3, earlystop_delta=0.5):
        """ Initializing all neccessary variables """
        super(ConditionalLogisticRegression, self).__init__()
        # try to use GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = lr
        self.epochs = int(epochs)
        self.groups_batch_size = int(groups_batch_size)
        self.l2_constant = l2_constant
        self.verbose = verbose
        self.earlystop_patience = earlystop_patience
        self.earlystop_delta= earlystop_delta


    def forward(self, X, strata):
        """ Function to compute the probability, overwritten from torch.nn.Module """
        y_hat = self.linear(X).squeeze()

        ix = 0
        for i, s in enumerate(strata):
            y_hat[ix:ix+s] = torch.nn.functional.softmax(y_hat[ix:ix+s], dim=0)
            ix += s

        return y_hat


    def neg_log_likelihood(self, y_pred, y_true):
        """ The negative log-likelihood function to optimize """
        # divide by number of samples N, for mini-batch gradient descent
        return -(torch.sum(y_true * torch.log(y_pred))) / self.groups_batch_size


    def fit(self, X, y, strata):
        """ Train the model using the provided data """
        # get X,y,strata in right format
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        if not isinstance(strata, pd.DataFrame):
            strata = pd.DataFrame(strata)
        strata.reset_index(drop=True, inplace=True)

        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device).squeeze()
        self.strata_list = list(strata.groupby(strata.squeeze()).groups.values())
        # check if strata is in correct shape
        if strata.shape[1] != 1:
            raise ValueError(f"strata has to be one-dimensional, got {strata.shape[1]}")
        # check if y is in correct shape
        if len(self.y.shape) > 1 and self.y.shape[1] != 1:
            raise ValueError("Only implemented for binary classification, 0 or 1")
        # set linear in loss function to correct shape
        input_size = self.X.shape[1]
        self.set_dimensions(input_size)
        # setup the optimizer 
        sgd = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.l2_constant)
        loss_list = []
        validation_loss_list = []
        # early stopper class load
        early_stopper = EarlyStopper(patience=self.earlystop_patience, min_delta=self.earlystop_delta)
        # mini-batch gradient descent loop
        for epoch in range(self.epochs+1):
            # shuffle data based on groups
            random.shuffle(self.strata_list)
            train_strata_list_idx = round(len(self.strata_list) * 0.8)
            train_strata_list = self.strata_list[:train_strata_list_idx]
            test_strata_list = self.strata_list[train_strata_list_idx:]
            # train on mini-batch
            batch_loss = []
            for batch in range(0, len(train_strata_list), self.groups_batch_size):
                # select based on batch
                strata_batch = train_strata_list[batch:batch+self.groups_batch_size]
                # need to flat the list of lists to be able to get the indices
                flat_strata_batch = [index for indices in strata_batch for index in indices]
                # get the length of each group/strata to sum in forward()
                strata_batch_len = [len(index) for index in strata_batch]

                X_batch = self.X[flat_strata_batch]
                y_batch = self.y[flat_strata_batch]

                # get probabilities
                y_pred = self.forward(X_batch, strata_batch_len)#.to(self.device)
                # negative log likelihood of predicted
                loss = self.neg_log_likelihood(y_pred, y_batch)
                # compute gradients and optimize
                sgd.zero_grad(set_to_none=True)
                loss.backward()
                sgd.step()
                # for the mean loss
                batch_loss.append(loss.item())
            # for logging, save loss for all epochs
            batch_loss = sum(batch_loss)/len(batch_loss)
            loss_list.append(batch_loss)
            if epoch % 10 == 0 and self.verbose:
                print(f"{epoch} : {sum(loss_list)/len(loss_list):.2f}")
            # early stopping if validation error begins to increase
            with torch.no_grad():
                flat_test_strata_list = [index for indices in test_strata_list for index in indices]
                X_test = self.X[flat_test_strata_list]
                y_test = self.y[flat_test_strata_list]
                test_strata_batch_len = [len(index) for index in test_strata_list]
                y_test_pred = self.forward(X_test, test_strata_batch_len)#.to(self.device)
                validation_loss = self.neg_log_likelihood(y_test_pred, y_test).item()
                validation_loss_list.append(validation_loss)
            if early_stopper.early_stop(validation_loss):
                print(f"Early stop epoch: {epoch}\nValidation loss: {sum(validation_loss_list)/len(validation_loss_list):.2f}"
                      f"\nTrain loss: {sum(loss_list)/len(loss_list):.2f}")
                break
        print(f"Train loss list: {loss_list}\nValidation loss list: {validation_loss_list}")


    def predict(self, X, strata=None, strata_list=None):
        """ Make predictions after fit and return binary """
        with torch.no_grad():
            if not strata_list:
                if not isinstance(strata, pd.DataFrame):
                    strata = pd.DataFrame(strata)
                strata.reset_index(drop=True, inplace=True)

                strata_list = list(strata.groupby(strata.squeeze()).groups.values())

            probabilities = self.predict_prob(X, strata, strata_list)
            for i in strata_list:
                max_prob_ix = probabilities[i].argmax()
                strata_ix = i[max_prob_ix]
                probabilities[i] = 0
                probabilities[strata_ix] = 1

            return probabilities.astype(int)


    def predict_prob(self, X, strata, strata_list=None):
        """ Make predictions after fit and return probabilities """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not isinstance(strata, pd.DataFrame):
            strata = pd.DataFrame(strata)
        strata.reset_index(drop=True, inplace=True)
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            #strata = strata.groupby(strata.squeeze()).groups.items()
            if not strata_list:
                strata_list = list(strata.groupby(strata.squeeze()).groups.values())
            flat_strata = [index for indices in strata_list for index in indices]
            strata_len = [len(index) for index in strata_list]

            original_index = torch.argsort(torch.tensor(flat_strata, device=self.device))
            X = X[flat_strata]

            return self.forward(X, strata_len)[original_index].float().squeeze().cpu().numpy()


    def get_coef(self):
        return [param.data.tolist() for param in self.parameters() if len(param.size()) > 1][0][0]


    def score(self):
        y_pred = self.predict(self.X, strata_list=self.strata_list)
        return np.mean(y_pred == self.y.float().squeeze().cpu().numpy())

    
    def set_dimensions(self, input_size):
        output_size = 1
        self.linear = torch.nn.Linear(input_size, output_size, bias=True, device=self.device)


