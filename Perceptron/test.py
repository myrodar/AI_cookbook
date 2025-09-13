import numpy as np
import matplotlib.pyplot as plt
from main import Perceptron   # import class from main.py

if __name__ == "__main__":
    np.random.seed(42)

    # linearly separable dataset
    n_per_class = 50
    class0 = np.random.randn(n_per_class, 2) * 0.6 + np.array([2.0, 2.0])
    class1 = np.random.randn(n_per_class, 2) * 0.6 + np.array([4.5, 4.0])

    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # Train perceptron
    clf = Perceptron(learning_rate=0.1, nb_iter=1001)
    clf.fit(X, y)

    # Report training accuracy
    pred = clf.predict(X)
    acc = (pred == y).mean()
    print(f"Training accuracy: {acc*100:.1f}% | epochs used: {clf.nb_iter}")

    # Plot points
    plt.figure(figsize=(6, 6))
    plt.scatter(class0[:, 0], class0[:, 1], label="Class 0", alpha=0.8)
    plt.scatter(class1[:, 0], class1[:, 1], label="Class 1", alpha=0.8)

    # Plot decision regions (background)
    h = 0.02
    x_min_bg, x_max_bg = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min_bg, y_max_bg = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x_min_bg, x_max_bg, h),
                         np.arange(y_min_bg, y_max_bg, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, zz, alpha=0.15, levels=[-0.5, 0.5, 1.5])

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    xs = np.linspace(x_min, x_max, 200)
    ys = clf.decision_boundary(xs)
    plt.plot(xs, ys, label="Decision boundary", linewidth=2)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Perceptron on a Simple 2D Dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()