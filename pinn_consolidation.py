"""PINN for 1D consolidation (Zhang, Chen & Yang, Sec. 4.1), dimensionless form:
    p_t - p_xx = 0,  0<x<1, 0<t<=0.5
    p(0,t) = 0,  p_x(1,t) = 0,  p(x,0) = 1
Output transform p = x * NN(x,t) enforces p(0,t)=0 exactly.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

TMAX = 0.5
EPOCHS_ADAM = 20000   # paper: 50000, then L-BFGS
LR = 1e-3


def analytic(x, t, terms=200):
    p = np.zeros_like(x)
    for m in range(1, 2 * terms, 2):
        p += 4 / (m * np.pi) * np.sin(m * np.pi * x / 2) * np.exp(-(m ** 2) * np.pi ** 2 * t / 4)
    return p


T = lambda a, g=False: torch.tensor(a, dtype=torch.float32, requires_grad=g).reshape(-1, 1)

# training points: 500 residual, 250 initial-condition (t=0), 125 Neumann (x=1)
xf, tf = T(np.random.rand(500), True), T(np.random.rand(500) * TMAX, True)
xi, ti = T(np.random.rand(250)), T(np.zeros(250))
xb, tb = T(np.ones(125), True), T(np.random.rand(125) * TMAX)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, d = [], 2
        for _ in range(5):
            layers += [nn.Linear(d, 50), nn.Tanh()]
            d = 50
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)

    def forward(self, x, t):
        return x * self.net(torch.cat([x, t], dim=1))


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


model = PINN()
mse = nn.MSELoss()


def loss_fn():
    p = model(xf, tf)
    p_t, p_x = grad(p, tf), grad(p, xf)
    p_xx = grad(p_x, xf)
    L_f = mse(p_t - p_xx, torch.zeros_like(p))
    L_i = mse(model(xi, ti), torch.ones_like(xi))
    pb = model(xb, tb)
    L_b = mse(grad(pb, xb), torch.zeros_like(pb))
    return 0.25 * L_f + 0.25 * L_i + 0.25 * L_b


opt = torch.optim.Adam(model.parameters(), lr=LR)
hist = []
for ep in range(EPOCHS_ADAM + 1):
    opt.zero_grad()
    loss = loss_fn()
    loss.backward()
    opt.step()
    hist.append(loss.item())
    if ep % 2000 == 0:
        print(f"Adam ep {ep:6d} loss {loss.item():.3e}")

# optional L-BFGS refinement (as in the paper)
lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=3000, tolerance_grad=1e-9,
                          tolerance_change=1e-12, history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    l = loss_fn()
    l.backward()
    return l


lbfgs.step(closure)
print("After L-BFGS loss:", loss_fn().item())

# ---- evaluation on a grid ----
xs, ts = np.linspace(0, 1, 101), np.linspace(1e-3, TMAX, 101)
X, TT = np.meshgrid(xs, ts)
with torch.no_grad():
    P = model(T(X.ravel()), T(TT.ravel())).numpy().reshape(X.shape)
Pref = analytic(X, TT)
err = np.abs(P - Pref)
print(f"Mean squared error: {np.mean((P - Pref) ** 2):.3e}   max abs error: {err.max():.3e}")

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, Z, ttl in zip(ax[:3], [Pref, P, err], ["Reference", "PINN", "Absolute error"]):
    im = a.pcolormesh(X, TT, Z, shading="auto", cmap="jet")
    a.set(title=ttl, xlabel="x", ylabel="t")
    plt.colorbar(im, ax=a)
ax[3].semilogy(hist)
ax[3].set(title="Adam loss history", xlabel="iteration")
plt.tight_layout()
plt.savefig("pinn_consolidation_results.png", dpi=150)
plt.show()

mask = TT >= 0.02
print(f"Max abs error (t >= 0.02): {err[mask].max():.3e}")
print(f"MSE (t >= 0.02): {np.mean((P[mask] - Pref[mask]) ** 2):.3e}")
