import pickle
import matplotlib.pyplot as plt
import numpy as np

from beegenn.protocol import Protocol

# any protocol instance will do
with open("../outputs/sim3/protocol.pickle", "rb") as f:
    protocol: Protocol = pickle.load(f)

fig, axs = plt.subplots(2, 4, layout="constrained", figsize=(18, 9))
for i, inhibition in enumerate([True, False]):
    for j, corr_type in enumerate(["homogeneous", "correlation", "covariance", "disconnected"]):
        ax = axs[i][j]
        matrix = protocol._generate_inhibitory_connectivity(corr_type, inhibition)
        #matrix = np.repeat(matrix, repeats=2, axis=0 )
        #matrix = np.repeat(matrix, repeats=4, axis=1 )
        image = ax.imshow(matrix, cmap="plasma", vmin=-0.3, vmax=0.3)
        ax.set_title(f"{corr_type} {'normal' if inhibition else 'self-inhibited'}")

    cbar = fig.colorbar(image, ax=ax)

plt.show()
plt.savefig("Connectivities.png")
