import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# -----------------------------
# Neural network layout
# -----------------------------

layers = [4, 6, 3]  # input, hidden, output
positions = []

for layer_index, number_of_nodes in enumerate(layers):
    x = layer_index
    y_values = np.linspace(
        -(number_of_nodes - 1) / 2,
        (number_of_nodes - 1) / 2,
        number_of_nodes
    )
    positions.append([(x, y) for y in y_values])


# -----------------------------
# Fake weights for visual effect
# -----------------------------

rng = np.random.default_rng(3)

weights = [
    rng.uniform(0.2, 1.0, size=(layers[i], layers[i + 1]))
    for i in range(len(layers) - 1)
]


# -----------------------------
# Animation helper
# -----------------------------

def pulse(frame, center, width=12):
    """
    Smooth activation pulse.
    Returns a value between 0 and 1.
    """
    return np.exp(-((frame - center) / width) ** 2)


# -----------------------------
# Figure setup
# -----------------------------

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect("equal")
ax.axis("off")

ax.set_xlim(-0.7, len(layers) - 0.3)
ax.set_ylim(-3.5, 3.5)

ax.set_title(
    "Neural Network Activation Flow",
    fontsize=18,
    pad=20
)

equation_text = ax.text(
    1,
    3.1,
    r"$a^{(l+1)} = \sigma(W^{(l)}a^{(l)} + b^{(l)})$",
    ha="center",
    va="center",
    fontsize=18
)


# -----------------------------
# Draw connections
# -----------------------------

connection_lines = []

for layer_index in range(len(layers) - 1):
    layer_connections = []

    for i, (x1, y1) in enumerate(positions[layer_index]):
        row = []

        for j, (x2, y2) in enumerate(positions[layer_index + 1]):
            line, = ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=0.7,
                alpha=0.15
            )
            row.append(line)

        layer_connections.append(row)

    connection_lines.append(layer_connections)


# -----------------------------
# Draw nodes
# -----------------------------

node_circles = []

for layer_index, layer_positions in enumerate(positions):
    layer_nodes = []

    for x, y in layer_positions:
        circle = Circle(
            (x, y),
            radius=0.14,
            fill=False,
            linewidth=2,
            alpha=0.5
        )
        ax.add_patch(circle)
        layer_nodes.append(circle)

    node_circles.append(layer_nodes)


# -----------------------------
# Labels
# -----------------------------

labels = ["Input Layer", "Hidden Layer", "Output Layer"]

for x, label in enumerate(labels):
    ax.text(
        x,
        -3.15,
        label,
        ha="center",
        va="center",
        fontsize=13
    )


# -----------------------------
# Animation update
# -----------------------------

def update(frame):
    # Node activation timing
    input_activation = pulse(frame, 20)
    hidden_activation = pulse(frame, 55)
    output_activation = pulse(frame, 90)

    layer_activations = [
        input_activation,
        hidden_activation,
        output_activation
    ]

    # Connection activation timing
    input_to_hidden_activation = pulse(frame, 38)
    hidden_to_output_activation = pulse(frame, 73)

    connection_activations = [
        input_to_hidden_activation,
        hidden_to_output_activation
    ]

    # Animate nodes
    for layer_index, nodes in enumerate(node_circles):
        activation = layer_activations[layer_index]

        for node_index, circle in enumerate(nodes):
            little_wave = 0.75 + 0.25 * np.sin(frame * 0.12 + node_index)

            circle.set_radius(0.14 + 0.12 * activation * little_wave)
            circle.set_alpha(0.25 + 0.75 * activation)
            circle.set_linewidth(1.5 + 3.5 * activation)

    # Animate connections
    for layer_index, layer_connections in enumerate(connection_lines):
        activation = connection_activations[layer_index]

        for i, row in enumerate(layer_connections):
            for j, line in enumerate(row):
                weight = weights[layer_index][i, j]

                line.set_alpha(0.08 + 0.75 * activation * weight)
                line.set_linewidth(0.6 + 3.0 * activation * weight)

    return []


# -----------------------------
# Run animation
# -----------------------------

animation = FuncAnimation(
    fig,
    update,
    frames=120,
    interval=40,
    blit=False,
    repeat=True
)

plt.show()