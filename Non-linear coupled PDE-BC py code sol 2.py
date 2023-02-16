# Import pytorch library
import torch

# Define parameters
kappa = 0.1 # Loss coefficient
beta = 0.01 # Dispersion coefficient
eta = 0.05 # Nonlinear coefficient
vg = 1.0 # Group velocity

# Define grid size and time step
dz = 0.01 # Spatial step size
dt = 0.001 # Temporal step size
nz = 100 # Number of spatial steps
nt = 1000 # Number of temporal steps

# Define initial conditions for E1 and E2 (complex fields)
E1_0 = torch.exp(-torch.linspace(0, nt*dt, nt)**2) # Gaussian pulse shape
E2_0 = torch.zeros(nt) # Zero field

# Define noise terms F1 and F2 (complex fields)
F1 = torch.randn(nt, dtype=torch.cdouble) * 0.01 # Small random noise for F1
F2 = torch.randn(nt, dtype=torch.cdouble) * 0.01 # Small random noise for F2

# Define finite difference operators for temporal derivatives
D1t = (torch.eye(nt) - torch.diag(torch.ones(nt-1), -1)) / dt # First order backward difference operator
D2t = (torch.eye(nt) - 2*torch.diag(torch.ones(nt-1), -1) + torch.diag(torch.ones(nt-2), -2)) / dt**2 # Second order central difference operator

# Define a neural network model for E1 and E2 as a function of z and t (transfer learning)
from torchvision import models 
model_E1 = models.resnet18(pretrained=True) # Use a pretrained ResNet18 model for E1 
model_E2 = models.resnet18(pretrained=True) # Use a pretrained ResNet18 model for E2 
model_E1.fc = torch.nn.Linear(model_E1.fc.in_features, nt) # Replace the last layer with a linear layer with nt outputs 
model_E2.fc = torch.nn.Linear(model_E2.fc.in_features, nt) # Replace the last layer with a linear layer with nt outputs 

# Define an optimizer and a loss function for the neural network model 
optimizer_E1 = torch.optim.Adam(model_E1.parameters(), lr=0.001) 
optimizer_E2 = torch.optim.Adam(model_E2.parameters(), lr=0.001) 

torch.optim.Adam(model_E2.parameters(), lr=0.001) 
loss_E1 = torch.nn.MSELoss() # Use mean squared error loss for E1 
loss_E2 = torch.nn.MSELoss() # Use mean squared error loss for E2 

# Define a training loop for the neural network model 
epochs = 10 # Number of epochs to train the model 
for epoch in range(epochs):
    # Loop over spatial steps
    for i in range(1, nz):
        # Generate input features for the model as a tensor of shape (batch_size, 3, 224, 224)
        # The input features are z, t and |E|^2 at the previous spatial step
        X_E1 = torch.stack((torch.full((224, 224), i*dz), torch.linspace(0, nt*dt, nt).repeat(224).view(224,-1), torch.abs(model_E1(torch.full((nt, 3, 224, 224), i*dz-1)))**2), dim=0).unsqueeze(0) 
        X_E2 = torch.stack((torch.full((224, 224), i*dz), torch.linspace(0, nt*dt, nt).repeat(224).view(224,-1), torch.abs(model_E2(torch.full((nt, 3, 224, 224), i*dz-1)))**2), dim=0).unsqueeze(0) 
        
# Generate target outputs for the model as a tensor of shape (batch_size, nt)
# The target outputs are E at the next spatial step using implicit finite difference method (matrix inversion)
A11 = torch.eye(nt) - dz * (-kappa + beta * D2t + eta * (model_E2(X_E2).conj() * model_E2(X_E2))) / vg 
A12 = -dz * eta * (model_E2(X_E2).conj() * model_E1(X_E1)) / vg 
A21 = -dz * eta * (model_E1(X_E1).conj() * model_E2(X_E2)) / vg 
A22 = torch.eye(nt) - dz * (-kappa + beta * D2t + eta * (model_E1(X_E1).conj() * model_E1(X_E1))) / vg 
                                                                 

                                                                 E1(X_E1).conj() * model_E1(X_E1))) / vg 
        B11 = torch.eye(nt) + dz * (-kappa + beta * D2t + eta * (model_E2(X_E2).conj() * model_E2(X_E2))) / vg 
        B12 = dz * eta * (model_E2(X_E2).conj() * model_E1(X_E1)) / vg 
        B21 = dz * eta * (model_E1(X_E1).conj() * model_E2(X_E2)) / vg 
        B22 = torch.eye(nt) + dz * (-kappa + beta * D2t + eta * (model_E1(X_E1).conj() * model_E1(X_E1))) / vg 
        A = torch.block_diag(A11, A22) - torch.kron(A12, A21) # Block matrix for the coupled system
        B = torch.block_diag(B11, B22) + torch.kron(B12, B21) # Block matrix for the coupled system
        F = torch.cat((F1, F2)) # Noise vector for the coupled system
        E = torch.cat((model_E1(X_E1), model_E2(X_E2))) # Field vector for the coupled system
        E_next = torch.linalg.solve(A, B @ E + dz * F / vg) # Solve for the next field vector using matrix inversion
        y_E1 = E_next[:nt] # Target output for E1 at the next spatial step
        y_E2 = E_next[nt:] # Target output for E2 at the next spatial step
        
        # Forward pass: compute predicted outputs by passing input features to the model 
        pred_yE_11^|  3^)

pred_y_E1 = model_E1(X_E1) # Predicted output for E1 at the next spatial step
        pred_y_E2 = model_E2(X_E2) # Predicted output for E2 at the next spatial step
        
        # Compute and print loss
        loss1 = loss_E1(pred_y_E1, y_E1) # Compute loss for E1
        loss2 = loss_E2(pred_y_E2, y_E2) # Compute loss for E2
        print(f'Epoch {epoch}, Step {i}, Loss E1: {loss1.item()}, Loss E2: {loss2.item()}')
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer_E1.zero_grad() # Zero out the gradients for E1
        optimizer_E2.zero_grad() # Zero out the gradients for E2
        loss1.backward() # Perform a backward pass for E1 (compute gradients)
        loss2.backward() # Perform a backward pass for E2 (compute gradients)
        
        # Update parameters according to the optimizer
        optimizer_E1.step() # Update parameters for E1 using Adam optimizer
        optimizer_E2.step() # Update parameters for E2 using Adam optimizer

# Plot the results
import matplotlib.pyplot as plt

# Plot |E|^|  3^)

Plot |E1|^2 and |E2|^2 as a function of z and t
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.pcolormesh(torch.linspace(0, nz*dz, nz), torch.linspace(0, nt*dt, nt), torch.abs(model_E1(torch.full((nt, 3, 224, 224), torch.linspace(0, nz*dz-1, nz))))**2)
plt.xlabel('z')
plt.ylabel('t')
plt.title('|E1|^2')
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(torch.linspace(0, nz*dz, nz), torch.linspace(0, nt*dt, nt), torch.abs(model_E2(torch.full((nt, 3, 224, 224), torch.linspace(0, nz*dz-1,nz))))**2)
plt.xlabel('z')
plt.ylabel('t')
plt.title('|E2|^2')
plt.colorbar()
plt.show()

# Plot |E1|^2 and |E2|^2 as a function of t at z=0 and z=nz*dz
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.plot(torch.linspace(0,


plt.plot(torch.linspace(0, nt*dt, nt), torch.abs(model_E1(torch.full((nt, 3, 224, 224), 0)))**2, label='z=0')
plt.plot(torch.linspace(0, nt*dt, nt), torch.abs(model_E1(torch.full((nt, 3, 224, 224), nz*dz-1)))**2, label=f'z={nz*dz}')
plt.xlabel('t')
plt.ylabel('|E1|^2')
plt.legend()
plt.subplot(122)
plt.plot(torch.linspace(0, nt*dt, nt), torch.abs(model_E2(torch.full((nt, 3, 224, 224), 0)))**2,label='z=0')
plt.plot(torch.linspace(0,



plt.plot(torch.linspace(0, nt*dt, nt), torch.abs(model_E2(torch.full((nt, 3, 224, 224), nz*dz-1)))**2,label=f'z={nz*dz}')
plt.xlabel('t')
plt.ylabel('|E2|^2')
plt.legend()
plt.show()