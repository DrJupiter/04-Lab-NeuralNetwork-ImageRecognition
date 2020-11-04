import torch
import numpy as np
import matplotlib.pyplot as plt



# Define weird function
def weird_fun(x):
    return np.sin(1/x)

# Reset random seed
np.random.seed(1)

# Set data parameters
N = 50 # Number of observations
s = 0.02 # Noise standard deviation

# Create training set
x_train = np.sort(np.random.rand(N)*2-1)
y_train = weird_fun(x_train) + s*np.random.randn(N)

# Create test set
x_test = np.sort(np.random.rand(N)*2-1)
y_test = weird_fun(x_test) + s*np.random.randn(N)

# Plot training data
plt.plot(x_train, y_train, '.');



# H is hidden dimension
H = 5

# Device to use for computations
device = torch.device('cpu')
#device = torch.device('cuda')

# Create Tensors to hold inputs and outputs
x = torch.tensor(np.expand_dims(x_train,1), dtype=torch.float32, device=device)
y = torch.tensor(np.expand_dims(y_train,1), dtype=torch.float32, device=device)

# Manually set random seed
#torch.manual_seed(1)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(1, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, 1),
)
model.to(device)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Number of iterations
T = 5000
# Allocate space for loss
Loss = np.zeros(T)

for t in range(T):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and save loss.
    loss = loss_fn(y_pred, y)
    Loss[t] = loss.item()

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()    

plt.plot(Loss)
plt.grid(True);

# Plot training data and fit
plt.subplot(121)
plt.plot(x_train, y_train, 'g.');
plt.plot(x_train, y_pred.cpu().detach().numpy(), 'r.')
train_error = loss_fn(y_pred, y).item()
plt.title('Training error: {:.2f}'.format(train_error))
# Plot test data and fit
x_t = torch.tensor(np.expand_dims(x_test,1), dtype=torch.float32, device=device)
y_t = torch.tensor(np.expand_dims(y_test,1), dtype=torch.float32, device=device)
y_t_pred = model(x_t)
x_all = np.linspace(-1,1,1000)
x_all_t = torch.tensor(np.expand_dims(x_all,1), dtype=torch.float32, device=device)
y_all_t = model(x_all_t)
plt.subplot(122)
plt.plot(x_all, y_all_t.cpu().detach().numpy(), 'r-');
plt.plot(x_test, y_t_pred.cpu().detach().numpy(), 'r.')
plt.plot(x_test, y_test, 'b.');
test_error = loss_fn(y_t_pred, y_t).item()
plt.title('Test error: {:.2f}'.format(test_error));