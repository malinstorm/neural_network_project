import numpy as np

X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])

y = np.array([[0],[1],[1],[0]])

np.random.seed(13)

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))


# Aktiveringsfunksjon: gjør verdiene om til tall mellom 0 og 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Hvor raskt vektene skal endres
learning_rate = 0.5

# Tren nettverket 10 000 ganger
for epoch in range(35000):

    # Forward pass
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    output = sigmoid(z2)

    # Beregn loss
    loss = np.mean((output - y) ** 2)

    # Backpropagation: feil i output-laget
    d_output = 2 * (output - y) / len(y)
    d_z2 = d_output * output * (1 - output)

    # Gradienter for W2 og b2
    d_W2 = a1.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    # Send feilen bakover til hidden layer
    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * a1 * (1 - a1)

    # Gradienter for W1 og b1
    d_W1 = X.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Oppdater vekter og bias
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2

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
print("\nW1:")
print(W1)

print("\nb1:")
print(b1)

print("\nW2:")
print(W2)

print("\nb2:")
print(b2)