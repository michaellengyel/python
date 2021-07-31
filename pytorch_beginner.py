import torch
import torch.nn as nn
import numpy as np


def main():

    print("pytorch")

    print("pytorch is available:", torch.cuda.is_available())

    print(torch.empty(2, 3))
    print(torch.rand(3, 2))
    print(torch.zeros(2, 2))
    print(torch.ones(2, 2))
    print(torch.tensor([2.4, 9, 0, 0.3]))

    print(torch.ones(2, 2, dtype=torch.int).dtype)
    print(torch.ones(2, 2, dtype=torch.float).dtype)
    print(torch.ones(2, 2, dtype=torch.bool).dtype)

    print(torch.ones(2, 3).size())

    print("pytorch operations:")

    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    print(x)
    print(y)

    # Addition (element wise)
    print(x + y)
    print(torch.add(x, y))
    # Inplace edition
    y.add_(x)
    print(y)

    # Subtraction (element wise)
    print(x - y)
    print(torch.sub(x, y))
    # Inplace subtraction
    y.sub_(x)
    print(y)

    # Multiplication (element wise)
    print(x * y)
    print(torch.mul(x, y))
    # Inplace multiplication
    y.mul_(x)
    print(y)

    # Division (element wise)
    print(x / x)
    print(torch.div(x, x))
    # Inplace multiplication
    x.div_(x)
    print(x)

    # Matrix multiplication
    print(x @ y)
    print(torch.matmul(x, y))

    # Slicing tensors
    x = torch.rand(5, 3)
    print(x)
    print(x[:, 0])
    print(x[1, :])

    # Get the actual value of a element in a tensor
    print(x[0, 0].item())

    # Reshaping a tensor
    x = torch.rand(4, 4)
    print(x)
    print(x.size())
    y = x.view(2, 8)
    print(y)
    print(y.size())

    # Converting from torch to numpy

    # Take note, that these two variables share the same memory location.
    # Changing one will change the other.

    a = torch.ones(5)
    print(a)
    print(type(a))
    b = a.numpy()
    print(b)
    print(type(b))

    # Converting from numpy to torch
    a = np.ones(5)
    print(a)
    print(type(a))
    b = torch.ones(5)
    print(b)
    print(type(b))

    # Moving tensors between cpu and gpu

    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        cpu_device = torch.device("cpu")
        x = torch.ones(5, device=gpu_device)  # Creates a tensor and puts it on the GPU
        y = torch.ones(5)
        y = y.to(gpu_device)  # Creates a tensor on CPU side and then moves it to the GPU
        z = x + y  # Operation performed on the GPU
        z = z.to(cpu_device)

    # Requires Gradient
    x = torch.ones(5, requires_grad=True)
    print(x)

    # Gradient calculation with Autograd
    x = torch.randn(3, requires_grad=True)
    print(x)

    # Computational graph created by operation
    y = x + 2
    print(y)
    z = y*y*2
    print(z)

    v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
    z.backward(v)
    print(x.grad)

    # Preventing the tracking of gradients
    # x.requires_grad_(False)
    # x.detach()
    # with torch.no_grad():
        # do operations

    # When calling the backward() function, the gradient for this tensor
    # will be accumulated into the .grad attribute, so the values will be
    # summed up.

    weights = torch.ones(4, requires_grad=True)

    for epoch in range(3):
        model_output = (weights*3).sum()
        model_output.backward()
        print(weights.grad)
        weights.grad.zero_()  # Very important line, if removed, grads accumulate

    # If using one of the default optimizer
    weights = torch.ones(4, requires_grad=True)
    optimizer = torch.optim.SGD([weights], lr=0.01)
    optimizer.step()
    optimizer.zero_grad()  # This does the same for default optimizers

    # Summery:
    # x.backward() calculates the gradients
    # weights.grad.zero_() needs to be called to empty the gradients for the net itr

    # Calculating the loss of a computational graph
    x = torch.tensor(1.0)
    y = torch.tensor(2.0)

    w = torch.tensor(1.0, requires_grad=True)

    # Forward pass and compute the loss
    y_hat = w * x;
    loss = (y_hat - y)**2

    print(loss)

    # Backward pass
    loss.backward()
    print(w.grad)

    # Update weights
    # Perform next forward and backward pass
    # ...

    # Machine learning algorithm from scratch only using numpy
    # f = w * x
    # f = 2 * x

    X = np.array([1, 2, 3, 4])
    Y = np.array([2, 4, 6, 8])

    w = 3330.0

    # model prediction
    def forward(x):
        return w * x

    # loss = MSR (in case of linear regression)
    def loss(y, y_predicted):
        return ((y_predicted - y)**2).mean()

    # gradient
    # MSE = 1/N(w*x - y)**2
    # dJ/dw = 1/N 2x (w*x-y)

    def gradient(x, y, y_predicted):
        return np.dot(2*x, y_predicted-y)

    # Training
    learning_rate = 0.01
    n_iters = 20

    for epoch in range(n_iters):
        # prediction = forward pass
        y_pred = forward(X)

        # loss
        l = loss(Y, y_pred)

        # gradients
        dw = gradient(X, Y, y_pred)

        # update weights
        w -= learning_rate * dw

        if epoch % 1 == 0:
            print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

    print(f'Prediction after training: f(x) = {forward(5):.3f}')

    # Machine learning algorithm from scratch only using torch
    # f = w * x
    # f = 2 * x

    X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

    w = torch.tensor(3330.0, dtype=torch.float32, requires_grad=True)

    # model prediction
    def forward(x):
        return w * x

    # loss = MSR (in case of linear regression)
    def loss(y, y_predicted):
        return ((y_predicted - y) ** 2).mean()

    # Training
    learning_rate = 0.01
    n_iters = 100

    for epoch in range(n_iters):
        # prediction = forward pass
        y_pred = forward(X)

        # loss
        l = loss(Y, y_pred)

        # gradients (backward pass)
        # dw = gradient(X, Y, y_pred)
        l.backward()  # dl/dw

        # update weights
        with torch.no_grad():
            w -= learning_rate * w.grad

        # Zero the gradients
        w.grad.zero_()

        if epoch % 1 == 0:
            print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

    print(f'Prediction after training: f(x) = {forward(5):.3f}')

    # Elements pytorch machine learning algorithm

    # 1. Design model (input, output size, forward pass)
    # 2. Construct loss and optimizer
    # 3. Training loop
    #    - Forward pass: Compute prediction
    #    - Backward pass: Compute gradiants
    #    - Update weights

    print("Using more pytorch functions:")

    X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
    Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

    x_test = torch.tensor([5], dtype=torch.float32)

    x_samples, n_features = X.shape
    print(x_samples, n_features)

    input_size = n_features
    output_size = n_features

    model = nn.Linear(input_size, output_size)

    print(f'Prediction before training: f(5) = {model(x_test).item():.3f}')

    # Training
    learning_rate = 0.01
    n_iters = 100

    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(n_iters):
        # prediction = forward pass
        y_pred = model(X)

        # loss
        l = loss(Y, y_pred)

        # gradients (backward pass)
        l.backward()  # dl/dw

        # update weights
        optimizer.step()

        # Zero the gradients
        optimizer.zero_grad()

        if epoch % 10 == 0:
            [w, b] = model.parameters()
            print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

    print(f'Prediction after training: f(5) = {model(x_test).item():.3f}')


if __name__ == '__main__':
    main()
