import numpy as np
import scipy.special


def generate_random_path(n):
    path = ['R'] # Initial step
    remaining_steps = ['R'] * (n - 1) + ['U'] * n
    np.random.shuffle(remaining_steps)
    path.extend(remaining_steps)
    return path


def is_valid_path(path):
    """Checks if a given path stays on or below the diagonal."""
    x, y = 0, 0  # Starting position
    for step in path:
        if step == 'R':
            x += 1
        else:  # 'U'
            y += 1
        if y > x:  # If we ever go above the diagonal, return False
            return False
    return True


def monte_carlo_estimate(n, N=10000):
    """Estimates c_n using Monte Carlo simulation."""
    valid_paths = sum(is_valid_path(generate_random_path(n)) for _ in range(N))
    # Compute |T_n| = (2n-1)! / (n! (n-1)!)
    T_n = scipy.special.comb(2 * n - 1, n, exact=True)
    # Monte Carlo estimate
    c_n_hat = T_n * (valid_paths / N)
    return c_n_hat


# Compute Monte Carlo estimates for n = 2, 3, 4
n_values = [2, 3, 4]
N = 10000  # Number of Monte Carlo samples
estimates = {n: monte_carlo_estimate(n, N) for n in n_values}

# Display results
import pandas as pd

df = pd.DataFrame.from_dict(estimates, orient='index', columns=['Monte Carlo Estimate'])
print(df)

t_n_values = dict([(n,scipy.special.comb(2*n-1,n,exact=True)) for n in n_values])
df = pd.DataFrame.from_dict(t_n_values, orient='index', columns=['|T_n|'])
print(df)