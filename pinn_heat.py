"""PINN for 2D steady-state heat conduction (Zhang, Chen & Yang, Sec. 4.2):
    T_xx + T_yy + 1 = 0 on [-1,1]^2,  T = 0 on the boundary.
The paper compares with a deal.II FEM solution; here a finite-difference
solution (scipy sparse) is used as the reference instead.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

torch.manual_seed(0)
np.random.seed(0)

EPOCHS = 20000   # paper: 50000 Adam + L-BFGS
LR = 5e-4


# ---------- reference: 5-point finite differences ----------
def fd_reference(n=101):
    xs = np.linspace(-1, 1, n)
    h = xs[1] - xs[0]
    k = n - 2  # interior points per side
    I = sp.identity(k)
    D = sp.diags([1, -2, 1], [-1, 0, 1], shape=(k, k))
    A = (sp.kron(I, D) + sp.kron(D, I)) / h ** 2
    u = spla.spsolve(A.tocsc(), -np.ones(k * k))  # T_xx + T_yy = -1
    Tfull = np.zeros((n, n))
    Tfull[1:-1, 1:-1] = u.reshape(k, k)
    return xs, Tfull


xs, T_ref = fd_reference()
X, Y = np.meshgrid(xs, xs, indexing="ij")  # T_ref[i, j] <-> (x_i, y_j)

# ---------- training points: 1500 interior, 500 boundary ----------
Tt = lambda a, g=False: torch.tensor(a, dtype=torch.float32, requires_grad=g).reshape(-1, 1)
xf, yf = Tt(np.random.uniform(-1, 1, 1500), True), Tt(np.random.uniform(-1, 1, 1500), True)

s = np.random.uniform(-1, 1, 500)
side = np.random.randint(0, 4, 500)
xb = np.where(side == 0, -1.0, np.where(side == 1, 1.0, s))
yb = np.where(side == 2, -1.0, np.where(side == 3, 1.0, s))
xb, yb = Tt(xb), Tt(yb)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, d = [], 2
        for _ in range(4):
            layers += [nn.Linear(d, 50), nn.Tanh()]
            d = 50
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def grad(f, x):
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]


model = PINN()
mse = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)
hist = []

for ep in range(EPOCHS + 1):
    opt.zero_grad()
    T = model(xf, yf)
    Txx = grad(grad(T, xf), xf)
    Tyy = grad(grad(T, yf), yf)
    L_f = mse(Txx + Tyy + 1.0, torch.zeros_like(T))
    L_b = mse(model(xb, yb), torch.zeros_like(xb))
    loss = 0.5 * L_f + 0.5 * L_b
    loss.backward()
    opt.step()
    hist.append(loss.item())
    if ep % 2000 == 0:
        print(f"ep {ep:6d} loss {loss.item():.3e}")

# ---------- evaluation ----------
with torch.no_grad():
    P = model(Tt(X.ravel()), Tt(Y.ravel())).numpy().reshape(X.shape)
err = np.abs(P - T_ref)
print(f"Mean squared error vs FD: {np.mean((P - T_ref) ** 2):.3e}   max abs error: {err.max():.3e}")

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, Z, ttl in zip(ax[:3], [T_ref, P, err], ["Reference (FD)", "PINN", "Absolute error"]):
    im = a.pcolormesh(X, Y, Z, shading="auto", cmap="jet")
    a.set(title=ttl, xlabel="x", ylabel="y")
    plt.colorbar(im, ax=a)
ax[3].semilogy(hist)
ax[3].set(title="Loss history", xlabel="iteration")
plt.tight_layout()
plt.savefig("pinn_heat_results.png", dpi=150)
plt.show()
