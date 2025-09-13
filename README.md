

# Binary Classifier Models

Welcome to the Binary Classifier Models repository. This library provides two simple, easy-to-understand implementations of fundamental machine learning algorithms for binary classification tasks, built with pure Python and NumPy.

## Models

- Decision Tree (`dt.py`): A depth-limited decision tree that splits features based on thresholding at 0.5.
- Perceptron (`perceptron.py`): A classic single-layer perceptron suitable for linearly separable data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/myrodar/AI_cookbook.git
   cd your-repo
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Getting Started

Coming soon!

### Decision Tree (`dt.py`)

- `DT(opts: dict)`  
  Initialize with options:  
  - `opts['maxDepth']`: Maximum depth of the tree.

- `train(X: np.ndarray, Y: Sequence[int])`  
  Build the decision tree from the data.

- `predict(x: np.ndarray) -> int`  
  Predict the label (`-1` or `1`) for a single sample.

- `__repr__()` / `displayTree()`  
  Print a human-readable representation of the tree.


## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes and push your branch.
4. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.