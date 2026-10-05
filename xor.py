import numpy as np

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])

y = np.array([[0],[1],[1],[0]])

np.random.seed(13)

Weight1 = np.random.randn(2, 4)
bias1 = np.zeros((1, 4))

Weight2 = np.random.randn(4, 1)
bias2 = np.zeros((1, 1))


# Aktiveringsfunksjon: gjør verdiene om til tall mellom 0 og 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Hvor raskt vektene skal endres
lr = 0.5

# Tren nettverket 10 000 ganger
for epoch in range(35000):

    # Forward pass
    z1 = X @ Weight1 + bias1
    a1 = sigmoid(z1)
    z2 = a1 @ Weight2 + bias2
    output = sigmoid(z2)

    # Beregn loss
    loss = np.mean((output - y) ** 2)

    # Backpropagation: feil i output-laget
    d_output = 2 * (output - y) / len(y)
    d_z2 = d_output * output * (1 - output)

    # Gradienter for Weight2 og bias2
    d_Weight2 = a1.T @ d_z2
    d_bias2 = np.sum(d_z2, axis=0, keepdims=True)

    # Send feilen bakover til hidden layer
    d_a1 = d_z2 @ Weight2.T
    d_z1 = d_a1 * a1 * (1 - a1)

    # Gradienter for Weight1 og bias1
    d_Weight1 = X.T @ d_z1
    d_bias1 = np.sum(d_z1, axis=0, keepdims=True)

    # Oppdater vekter og bias
    Weight1 -= lr * d_Weight1
    bias1 -= lr * d_bias1
    Weight2 -= lr * d_Weight2
    bias2 -= lr * d_bias2

    # Vis utviklingen hver 1000. runde
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}")

print("\nFerdige prediksjoner:")
print(output)

# Gjør sannsynlighetene om til 0 eller 1
predictions = (output >= 0.5).astype(int)

print("\nKlassifisering:")
print(predictions)

# Sammenlign med fasit
accuracy = np.mean(predictions == y)
print(f"\nAccuracy: {accuracy * 100:.0f}%")

# Se hva nettverket faktisk har lært
print("\nWeight1:")
print(Weight1)

print("\nbias1:")
print(bias1)

print("\nWeight2:")
print(Weight2)

print("\nbias2:")
print(bias2)
