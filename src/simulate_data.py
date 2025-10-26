import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def simulate_data(n_samples=50_000, test_size=0.2):
    """
    Simulates a dataset with X (predictors), T (treatment), Y (outcome) where T affects Y
    depending on X; then splits data into training and test sets.

    Parameters:
    n_samples (int): Number of samples to generate.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    None: Saves the generated dataset to CSV files.
    """
    # Set the random seed for reproducibility
    rs = 2024
    np.random.seed(rs)

    # Generate random values for columns X1, X2, X3, X4
    X1 = np.random.rand(n_samples)
    X2 = np.random.rand(n_samples)
    X3 = np.random.rand(n_samples)
    X4 = np.random.rand(n_samples)

    # Create column X5 to have a correlation of 0.3 with column X4
    noise = np.random.rand(n_samples)
    X5 = 0.3 * X4 + np.sqrt(1 - 0.3 ** 2) * noise

    # Generate binary columns X6 and X7 with a probability of 0.2 for 1
    X6 = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    X7 = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    # Generate binary column T with a 50% probability for 0 and 1
    T = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

    # Initialize column Y
    Y = np.zeros(n_samples, dtype=int)

    # Create column Y based on T and some X columns,
    # effectively creating treatment heterogeneities
    for i in range(n_samples):
        if T[i] == 1:
            prob_Y = 0.30 \
                     - 0.04 * (X1[i] + X2[i]) \
                     + 0.16 * (X3[i] * X4[i]) \
                     - 0.02 * X5[i] \
                     + 0.02 * X6[i]
        else:
            prob_Y = 0.30

        # Ensure the probability is within the range [0, 1]
        prob_Y = min(max(prob_Y, 0), 1)

        # Generate Y based on the calculated probability
        Y[i] = np.random.choice([0, 1], p=[1 - prob_Y, prob_Y])

    # Create a DataFrame
    df = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'X4': X4,
        'X5': X5,
        'X6': X6,
        'X7': X7,
        'T': T,
        'Y': Y
    })

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=rs)
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    print("Data simulation complete. Files saved to data/processed/train_data.csv and data/processed/test_data.csv")


if __name__ == "__main__":
    simulate_data()
