import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.environment import env_manager

env = env_manager.Environment()

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(env.env)

# Create a Rectangle patch
rect = patches.Rectangle((406,835),1800,200,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
