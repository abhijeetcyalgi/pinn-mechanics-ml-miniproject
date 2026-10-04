"""PINN inverse problem: identify damping c and stiffness k of
m*u'' + c*u' + k*u = 0, u(0)=1, u'(0)=1  (m = 1).
True values: c = 0.4, k = 4.0 (as in Zhang, Chen & Yang).
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

m, u0, v0 = 1.0, 1.0, 1.0
C_TRUE, K_TRUE = 0.4, 4.0
T = 15.0
EPOCHS = 30000
LR = 1e-3


def analytic(t, c=C_TRUE, k=K_TRUE):
    wn = np.sqrt(k / m)
    xi = c / (2 * m * wn)
    wd = wn * np.sqrt(1 - xi ** 2)
    return np.exp(-xi * wn * t) * (
        u0 * np.cos(wd * t) + (v0 + xi * wn * u0) / wd * np.sin(wd * t)
    )


# ---- data: 50 equispaced observations, 100 random collocation points ----
t_data = np.linspace(0, T, 50)
u_data = analytic(t_data)
t_f = np.random.uniform(0, T, 100)

to_t = lambda a, grad=False: torch.tensor(
    a, dtype=torch.float32, requires_grad=grad
).reshape(-1, 1)
td, ud = to_t(t_data), to_t(u_data)
tf = to_t(t_f, grad=True)
t0 = to_t([0.0], grad=True)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, d = [], 1
        for _ in range(3):
            layers += [nn.Linear(d, 32), nn.Tanh()]
            d = 32
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
        # unknown physical parameters, learned together with the weights
        self.c = nn.Parameter(torch.tensor(1.0))
        self.k = nn.Parameter(torch.tensor(1.0))

    def forward(self, t):
        return self.net(t / T)  # scale input to O(1)


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


model = PINN()
opt = torch.optim.Adam(model.parameters(), lr=LR)
mse = nn.MSELoss()
hist = {"loss": [], "c": [], "k": []}

for ep in range(EPOCHS + 1):
    opt.zero_grad()
    # ODE residual
    u = model(tf)
    ut = grad(u, tf)
    utt = grad(ut, tf)
    L_f = mse(m * utt + model.c * ut + model.k * u, torch.zeros_like(u))
    # initial conditions
    u_i = model(t0)
    L_b = mse(u_i, torch.tensor([[u0]])) + mse(grad(u_i, t0), torch.tensor([[v0]]))
    # observations
    L_d = mse(model(td), ud)
    loss = 0.25 * L_f + 0.25 * L_b + 0.25 * L_d
    loss.backward()
    opt.step()

    hist["loss"].append(loss.item())
    hist["c"].append(model.c.item())
    hist["k"].append(model.k.item())
    if ep % 2000 == 0:
        print(f"ep {ep:6d} loss {loss.item():.2e}  c={model.c.item():.4f}  k={model.k.item():.4f}")

print(f"\nFinal: c = {model.c.item():.5f} (true {C_TRUE}), k = {model.k.item():.5f} (true {K_TRUE})")

# ---- plots ----
tt = np.linspace(0, T, 500)
with torch.no_grad():
    pred = model(to_t(tt)).numpy().ravel()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(tt, analytic(tt), "k", label="True")
ax[0].plot(tt, pred, "r--", label="PINN")
ax[0].scatter(t_data, u_data, s=10, c="b", label="Observations")
ax[0].set(xlabel="t", ylabel="u", title="Solution")
ax[0].legend()
ax[1].plot(hist["c"], label="c")
ax[1].plot(hist["k"], label="k")
ax[1].axhline(C_TRUE, ls=":", c="gray")
ax[1].axhline(K_TRUE, ls=":", c="gray")
ax[1].set(xlabel="iteration", title="Parameter identification")
ax[1].legend()
ax[2].semilogy(hist["loss"])
ax[2].set(xlabel="iteration", title="Total loss")
plt.tight_layout()
plt.savefig("pinn_oscillator_results.png", dpi=150)
plt.show()
