import torch
import random
import pandas as pd

class ConditionalLogisticRegression(torch.nn.Module):
    """
    Conditional / Fixed-Effects Logistic Regression model implemented with PyTorch.
    """
    def __init__(self, X, y, strata, learning_rate=0.0001, max_iter=1000, groups_batch_size=5):
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
            print(f"strata has to be one-dimensional, got {strata.shape[1]}")
            exit()

        self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(y, dtype=torch.float32, device=self.device).squeeze()
        self.strata_list = list(strata.groupby(strata.squeeze()).groups.values())

        if len(self.y.shape) > 1 and self.y.shape[1] != 1:
            print("Only implemented for binary classification, 0 or 1")
            exit()

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.groups_batch_size = groups_batch_size

        input_size = self.X.shape[1]
        self.output_size = 1

        self.linear = torch.nn.Linear(input_size, self.output_size, bias=True, device=self.device)


    def forward(self, X, strata):
        """ Function to compute the probability, overwritten from torch.nn.Module """
        y_hat = self.linear(X).squeeze()
        # create matrix filled with -300 to get 0 when taking softmax (e^(-300) ≈ 0)
        y_hat_matrix = torch.full((len(strata), max(strata)), -300, dtype=torch.float32, device=self.device)
        ix = 0
        for i, s in enumerate(strata):
            y_hat_matrix[i,0:s] = y_hat[ix:ix+s]
            ix += s
        y_hat_matrix = torch.nn.functional.softmax(y_hat_matrix, dim=1)

        return y_hat_matrix[y_hat_matrix != 0]


    def neg_log_likelihood(self, y_pred, y_true):
        """ The negative log-likelihood function to optimize """
        # divide by number of samples N, for mini-batch gradient descent
        return -(torch.sum(y_true * torch.log(y_pred))) / self.groups_batch_size


    def fit(self):
        """ Train the model using the provided data """
        sgd = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

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

                #strata_batch = strata_shuffled[batch:batch+batch_size]
                # get probabilities
                y_pred = self.forward(X_batch, strata_batch_len).to(self.device)
                # negative log likelihood of predicted
                loss = self.neg_log_likelihood(y_pred, y_batch)
                # compute gradients and optimize
                sgd.zero_grad()
                loss.backward()
                sgd.step()

            #if epoch % 100 == 0:
            #    print(f"{epoch} : {loss:.2f}")


    def predict(self, X, strata):
        """ Make predictions after fit and return binary """
        with torch.no_grad():
            return (self.predict_prob(X, strata) > 0.5).float().squeeze().cpu().numpy()


    def predict_prob(self, X, strata):
        """ Make predictions after fit and return probabilities """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not isinstance(strata, pd.DataFrame):
            strata = pd.DataFrame(strata)

        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            strata_list = list(strata.groupby(strata.squeeze()).groups.values())
            #strata = strata.groupby(strata.squeeze()).groups.items()
            flat_strata = [index for indices in strata_list for index in indices]
            strata_len = [len(index) for index in strata_list]

            return self.forward(X, strata_len)


