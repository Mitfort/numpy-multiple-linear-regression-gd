"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    np.random.seed(seed)

    n:int = X.shape[0]

    if n == 0: return X, y

    idxs = np.ndarray = np.arange(n)

    np.random.shuffle(idxs)

    return X[idxs], y[idxs]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    n:int = len(y)

    n_train:int = int(n * train_frac)
    n_val:int = int(n * val_frac)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,y_test = X[n_train+n_val:], y[n_train+n_val:]

    return (X_train,y_train, X_val,y_val, X_test,y_test)

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)

    std[std == 0.0] = 1.0

    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    return (X-mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    n:int = X.shape[0]

    bias:np.ndarray = np.ones((n,1))

    res:np.ndarray = np.concat([bias,X], axis=1)

    return res

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    X = standardize_features(X,mean,std)

    X = add_bias_column(X)

    return X

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    n:int = len(y_true)

    return (1/n) * np.sum((y_pred - y_true)**2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    n:int = len(y_true)

    return (2/n) * X.T @ (y_pred - y_true)

# Step 10 - normal_equation
def normal_equation(X, y):
    A = X.T @ X
    b = X.T @ y 

    return np.linalg.pinv(A) @ b

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    
    weights = np.random.normal(0,0.01,size=n_features)

    return weights

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """

    preds = predict_linear(X,weights)

    grad = mse_gradient(X,y,preds)

    return weights - lr * grad

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """

    preds = predict_linear(X_train,weights)
    loss = mse_loss(y_train,preds)

    val_preds = predict_linear(X_val,weights)
    val_loss = mse_loss(y_val, val_preds)

    return (loss.item(), val_loss.item())

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    if val_loss < best_val_loss: 
        wait = 0
        best_weights = weights.copy()
        best_val_loss = val_loss

        return (best_val_loss, wait, best_weights, False)
    else:
        stop = False 
        wait += 1

        if wait >= patience: 
            stop = True 
        
        return (best_val_loss, wait, best_weights, stop)

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    weights = initialize_weights(n_features,seed)

    best_weights = weights.copy()

    best_val_loss = np.inf

    return {
        "weights": weights,
        "best_weights": best_weights,
        "best_val_loss": best_val_loss,
        "wait": 0,
        "train_losses": [],
        "val_losses": [],
        "stopped": False
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """

    weights = gd_step(X_train,y_train,state['weights'],lr)

    train_loss,val_loss = epoch_train_val_losses(X_train,y_train,X_val,y_val,weights)

    state['weights'] = weights
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)

    best_val_loss, wait, best_weights,stopped = update_early_stop_state(
        val_loss, state['best_val_loss'], state['wait'], weights, state['best_weights'], patience
    )

    state['best_val_loss'] = best_val_loss
    state['wait'] = wait 
    state['best_weights'] = best_weights
    state['stopped'] = stopped
    
    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    
    n_features:int = X_train.shape[1]

    state = init_training_state(n_features,seed=seed)

    for epoch in range(epochs):
        state = run_one_epoch(state,X_train,y_train,X_val,y_val,lr,patience)

        if state['stopped']: break 
    
    return (state['best_weights'], state['train_losses'], state['val_losses'])

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    
    return np.mean(np.abs(y_pred - y_true))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    
    return np.sqrt(np.mean((y_pred-y_true)**2))

# Step 20 - r_squared
def r_squared(y_true, y_pred):
    
    mean_y = np.mean(y_true)
    RSS = np.sum((y_true - y_pred)**2)
    TSS = np.sum((y_true-mean_y)**2)

    if TSS == 0: return np.nan

    R2 = 1 - RSS / TSS 

    return R2

# Step 21 - evaluate_regression
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    r2 = r_squared(y_true,y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2 
    }

# Step 22 - learning_curve_data
def learning_curve_data(train_losses, val_losses):

    if isinstance(train_losses,np.ndarray):
        train_losses = train_losses.tolist()
        
    if isinstance(val_losses, np.ndarray):
        val_losses = val_losses.tolist()
    

    idxs = [i+1 for i in range(len(train_losses))]

    return (idxs,train_losses,val_losses)

# Step 23 - weights_l2_distance
def weights_l2_distance(w_gd, w_closed):
    
    l2 = np.sqrt(np.sum((w_gd - w_closed)**2))

    return l2

# Step 24 - create_lr_model
def create_lr_model(learning_rate=0.01, epochs=1000, patience=50, seed=0):
    return {
        'learning_rate': learning_rate,
        'epochs': epochs,
        'patience': patience,
        'seed': seed,
        'weights': None,
        'normal_weights': None,
        'mean': None,
        'std': None,
        'train_losses': [],
        'val_losses': []
    }

# Step 25 - fit_lr_model
def fit_lr_model(model, X_train, y_train, X_val, y_val):
    mean,std = compute_feature_stats(X_train)

    X_train = prepare_design_matrix(X_train, mean, std)
    X_val = prepare_design_matrix(X_val, mean, std)

    weights, train_losses, val_losses = train_batch_gd(X_train,y_train, X_val, y_val, 
        model['learning_rate'], 
        model['epochs'],
        model['patience'],
        seed=model['seed']
    )

    normal_weights = normal_equation(X_train, y_train)

    model['weights'] = weights
    model['train_losses']  = train_losses
    model['val_losses'] = val_losses
    model['mean'] = mean
    model['std'] = std

    model['normal_weights'] = normal_weights  

    return model

# Step 26 - predict_lr_model
def predict_lr_model(model, X):
    X_desing = prepare_design_matrix(X, model['mean'], model['std'])

    preds = predict_linear(X_desing, model['weights'])

    return preds

# Step 27 - score_lr_model
import numpy as np
def score_lr_model(model, X, y):
    preds = predict_lr_model(model,X)

    return evaluate_regression(y,preds)

# Step 28 - compare_with_normal_equation
def compare_with_normal_equation(model):
    
    return weights_l2_distance(model['weights'], model['normal_weights'])

