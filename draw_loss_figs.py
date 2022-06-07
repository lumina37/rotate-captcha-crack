import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG

xx = np.linspace(-4, 4, 500, dtype=np.float32)
yy = np.linspace(0, 1, 500, dtype=np.float32)
x, y = np.meshgrid(xx, yy)
abs_diff = np.abs(y - x)

lambda_cos: float = CONFIG['train']['loss']['lambda_cos']
exponent: float = CONFIG['train']['loss']['exponent']

fig = plt.figure(figsize=(8, 8))
ax3 = plt.axes(projection='3d')
z = np.power(abs_diff, exponent) - (np.cos(abs_diff * (np.pi * 2)) - 1) * lambda_cos
ax3.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma')
ax3.set_xlabel('predict', fontsize='large')
ax3.set_ylabel('target', fontsize='large')
ax3.set_zlabel('loss', fontsize='large')
fig.savefig("figures/This-RotationLoss.png")

fig = plt.figure(figsize=(8, 8))
ax3 = plt.axes(projection='3d')
z = 0.5 - np.abs(abs_diff - 0.5)
ax3.plot_surface(x, y, z, rstride=1, cstride=1, cmap='plasma')
ax3.set_xlabel('predict', fontsize='large')
ax3.set_ylabel('target', fontsize='large')
ax3.set_zlabel('loss', fontsize='large')
fig.savefig("figures/RotNet-angle_error_regression.png")
