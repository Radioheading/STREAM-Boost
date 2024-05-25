import torch
import torch.nn as nn

class CustomKernel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # xavier initialization
        self.W = nn.Parameter(torch.randn(input_dim, output_dim).cuda())
        self.c = nn.Parameter(torch.Tensor([1.0]).cuda())

    def forward(self, X):
        WTW_I = self.W @ self.W.T + torch.eye(self.W.shape[0], device=X.device).cuda()
        return X @ WTW_I @ X.T + self.c

class MMDLoss(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        X = X.to(X.device)
        Y = Y.to(X.device)

        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        norm_W = torch.norm(self.kernel.W)
        loss = XX - 2 * XY + YY - torch.log(norm_W) - torch.log(self.kernel.c)
        return loss

# Example usage
input_dim = 2
output_dim = 3
batch_size = 2

X = torch.randn(batch_size, input_dim).cuda()
Y = torch.randn(batch_size, input_dim).cuda()

kernel = CustomKernel(input_dim, output_dim)
mmd_loss = MMDLoss(kernel)

optimizer = torch.optim.Adam([{'params': kernel.parameters()}], lr=0.01)

for i in range(1000):
    optimizer.zero_grad()
    loss = mmd_loss(X, Y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f'Iteration {i}, Loss: {loss.item()}')

# Retrieve the learned parameters
learned_W = kernel.W
learned_c = kernel.c
print('Learned W:\n', learned_W)
print('Learned c:', learned_c)
