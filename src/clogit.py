import torch
import random
import pandas as pd


class ConditionalLogisticRegression(torch.nn.Module):
    """
    Conditional / Fixed-Effects Logistic Regression model implemented with PyTorch.
    """
    def __init__(self, X, y, strata, learning_rate=0.00001, max_iter=100, groups_batch_size=1, regularization=None, regularization_constant=0.5):
        """ Initializing all neccessary variables """
        super(ConditionalLogisticRegression, self).__init__()
        # try to use GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        if not isinstance(strata, pd.DataFrame):
            strata = pd.DataFrame(strata)
        strata.reset_index(drop=True, inplace=True)

        if strata.shape[1] != 1:
            raise ValueError(f"strata has to be one-dimensional, got {strata.shape[1]}")

        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device).squeeze()
        self.strata_list = list(strata.groupby(strata.squeeze()).groups.values())

        if len(self.y.shape) > 1 and self.y.shape[1] != 1:
            raise ValueError("Only implemented for binary classification, 0 or 1")

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.groups_batch_size = groups_batch_size
        self.regularization_constant = regularization_constant

        input_size = self.X.shape[1]
        self.output_size = 1

        self.linear = torch.nn.Linear(input_size, self.output_size, bias=True, device=self.device)

        if regularization == "l1":
            self.regularization = lambda beta: torch.sum(torch.abs(beta)).to(self.device)
        elif regularization == "l2":
            self.regularization = lambda beta: torch.sum(torch.square(beta)).to(self.device)
        else:
            self.regularization = lambda beta: 0


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
        if self.regularization:
            weights = torch.cat([value.view(-1) for value in self.state_dict().values()])
        return -(torch.sum(y_true * torch.log(y_pred))) / self.groups_batch_size + self.regularization_constant * self.regularization(weights)


    def fit(self):
        """ Train the model using the provided data """
        sgd = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        loss_list = []

        for epoch in range(self.max_iter+1):
            # shuffle data based on groups
            random.shuffle(self.strata_list)
            # train on mini-batch
            for batch in range(0, len(self.strata_list), self.groups_batch_size):
                # select based on batch
                strata_batch = self.strata_list[batch:batch+self.groups_batch_size]
                # need to flat the list of lists to be able to get the indices
                flat_strata_batch = [index for indices in strata_batch for index in indices]
                # get the length of each group/strata to sum in forward()
                strata_batch_len = [len(index) for index in strata_batch]

                X_batch = self.X[flat_strata_batch]
                y_batch = self.y[flat_strata_batch]

                # get probabilities
                y_pred = self.forward(X_batch, strata_batch_len).to(self.device)
                # negative log likelihood of predicted
                loss = self.neg_log_likelihood(y_pred, y_batch)
                # compute gradients and optimize
                sgd.zero_grad()
                loss.backward()
                sgd.step()

            if epoch % 10 == 0:
                loss_list.append(loss.item())
                print(f"{epoch} : {sum(loss_list)/len(loss_list):.2f}")


    def predict(self, X, strata):
        """ Make predictions after fit and return binary """
        with torch.no_grad():
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

        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            #strata = strata.groupby(strata.squeeze()).groups.items()
            if not strata_list:
                strata_list = list(strata.groupby(strata.squeeze()).groups.values())
            flat_strata = [index for indices in strata_list for index in indices]
            strata_len = [len(index) for index in strata_list]

            original_index = torch.argsort(torch.tensor(flat_strata, device=self.device))
            X = X[flat_strata]

            return self.forward(X, strata_len)[original_index].float().squeeze().cpu().numpy()


