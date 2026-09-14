import numpy as np

np.random.seed(42)
learning_rate = 0.1

# Edit these arrays to change the data and dimensions.
X = np.array([
	[1., 2., -1.],
	[0., -1., 2.]
], dtype=np.float32)
# Ground-truth targets, with one row per example.
Y = np.array([
	[1., -0.5],
	[0.5, 2.]
], dtype=np.float32)

# Random parameters and predictions before training.
bound = 1 / np.sqrt(X.shape[1])
W = np.random.uniform(-bound, bound, size=(X.shape[1], Y.shape[1])).astype(np.float32)
b = np.random.uniform(-bound, bound, size=(1, Y.shape[1])).astype(np.float32)
Y_before = X @ W + b
loss_before = np.mean((Y_before - Y) ** 2)
initial_W, initial_b = W.copy(), b.copy()
sample_indices = np.random.randint(len(X), size=10)

def update(learning_rate):
	global W, b, Y_pred, loss, loss_history, step, batch_X, batch_Y, dW, db
	if not np.isfinite(learning_rate) or learning_rate < 0:
		raise ValueError("Use a finite, nonnegative learning rate")
	W, b = initial_W.copy(), initial_b.copy()
	loss_history = [loss_before]
	print(f"Update 00: MSE = {loss_before:.3f}")
	for step, index in enumerate(sample_indices):
		# Sample one example; compute its prediction and loss gradient.
		batch_X = X[index:index + 1]
		batch_Y = Y[index:index + 1]
		prediction = batch_X @ W + b
		dY_pred = 2 * (prediction - batch_Y) / prediction.size
		dW = batch_X.T @ dY_pred
		db = dY_pred.sum(axis=0, keepdims=True)

		# SGD: move parameters opposite the gradient.
		W -= learning_rate * dW
		b -= learning_rate * db

		# Evaluate all examples after the update.
		Y_pred = X @ W + b
		loss = np.mean((Y_pred - Y) ** 2)
		loss_history.append(loss)
		print(f"Update {step + 1:02d}: MSE = {loss:.3f}")
	return {"INPUTS": {"X": X, "Y": Y}, "OUTPUTS": {
		"Y_before": Y_before, "Y_pred": Y_pred, "W": W, "b": b,
	}}

# Display-only exports; the computations above do not depend on these.
PARAMETERS = {"learning_rate": learning_rate}
globals().update(update(**PARAMETERS))