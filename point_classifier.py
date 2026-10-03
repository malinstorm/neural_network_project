import numpy as np
import matplotlib.pyplot as plt

# Samme tilfeldige data hver gang vi kjører programmet
np.random.seed(13)

# Lag 1000 tilfeldige punkter med x- og y-koordinat
X = np.random.randn(1000, 2)

# Euklidsk distanse
distance = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)

# Punkter innenfor radius 1 blir klasse 1, resten klasse 0
y = (distance < 1).astype(int).reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Klasse 0:", np.sum(y == 0))
print("Klasse 1:", np.sum(y == 1))

# Bland dataene
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Del i 70 % trening, 15 % validering, 15 % test
train_end = int(len(X) * 0.70)
val_end = int(len(X) * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]



# Arkitektur: 2 inputs -> 8 hidden-nevroner -> 1 output
input_size = 2
hidden_size = 8
output_size = 1

# Initialiser vekter tilfeldig og bias til 0
W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))

# Sigmoid-aktivering
def sigmoid(x):
    return 1 / (1 + np.exp(-x))





# Treningsinnstillinger
learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):

    # Forward pass
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    output = sigmoid(z2)

    # Loss på treningsdata
    loss = np.mean((output - y_train) ** 2)

    # Backpropagation
    d_output = 2 * (output - y_train) / len(y_train)
    d_z2 = d_output * output * (1 - output)

    d_W2 = a1.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * a1 * (1 - a1)

    d_W1 = X_train.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Gradient descent
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: train loss = {loss:.4f}")



    # Forward pass på valideringsdata
z1_val = X_val @ W1 + b1
a1_val = sigmoid(z1_val)
z2_val = a1_val @ W2 + b2
output_val = sigmoid(z2_val)

# Gjør sannsynlighetene om til klasse 0 eller 1
pred_val = (output_val >= 0.5).astype(int)

# Accuracy på valideringssettet
val_accuracy = np.mean(pred_val == y_val)

print(f"\nValidation accuracy: {val_accuracy * 100:.2f}%")


# Finn valideringspunktene modellen klassifiserte feil
wrong = np.where(pred_val.flatten() != y_val.flatten())[0]

print("\nFeilklassifiserte punkter:")

for i in wrong:
    x1, x2 = X_val[i]
    distance = np.sqrt(x1**2 + x2**2)

    print(
        f"x=({x1:.3f}, {x2:.3f}) "
        f"distance={distance:.3f} "
        f"fasit={y_val[i, 0]} "
        f"prediksjon={output_val[i, 0]:.3f}"
    )

# Lag et rutenett av punkter over området
x1_range = np.linspace(-3, 3, 300)
x2_range = np.linspace(-3, 3, 300)
xx, yy = np.meshgrid(x1_range, x2_range)

# Gjør rutenettet om til datapunkter modellen kan lese
grid = np.c_[xx.ravel(), yy.ravel()]

# Kjør punktene gjennom det trente nettverket
z1_grid = grid @ W1 + b1
a1_grid = sigmoid(z1_grid)
z2_grid = a1_grid @ W2 + b2
grid_output = sigmoid(z2_grid)

# Gjør resultatet tilbake til samme form som rutenettet
grid_output = grid_output.reshape(xx.shape)

# Tegn modellens beslutningsgrense ved output = 0.5
plt.contour(xx, yy, grid_output, levels=[0.5])

# Tegn valideringspunktene
plt.scatter(
    X_val[:, 0],
    X_val[:, 1],
    c=y_val.flatten(),
    s=25
)

# Tegn den ekte sirkelgrensen
circle = plt.Circle((0, 0), 1, fill=False)
plt.gca().add_patch(circle)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("MLP decision boundary vs. true boundary")
plt.axis("equal")
plt.savefig("decision_boundary.png", dpi=300, bbox_inches="tight")
plt.show()
