import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def get_ate_w_ci(foc_data, y, t, z=1.96):
    """
    Calculates the Average Treatment Effect (ATE) with Confidence Intervals (CI).

    Parameters:
    foc_data (pd.DataFrame): Data containing outcome and treatment columns.
    y (str): Column name for the outcome variable.
    t (str): Column name for the treatment variable.
    z (float): Z-score for confidence interval calculation. Default is 1.96 for 95% CI.

    Returns:
    tuple: Coefficient, standard error, and confidence interval lower and upper bounds.
    """
    t_coeff = (np.sum((foc_data[t] - foc_data[t].mean()) * (foc_data[y] - foc_data[y].mean())) /
               np.sum((foc_data[t] - foc_data[t].mean()) ** 2))

    n = foc_data.shape[0]
    t_bar = foc_data[t].mean()
    beta1 = t_coeff
    beta0 = foc_data[y].mean() - beta1 * t_bar
    e = foc_data[y] - (beta0 + beta1 * foc_data[t])
    se = np.sqrt(((1 / (n - 2)) * np.sum(e ** 2)) / np.sum((foc_data[t] - t_bar) ** 2))
    t_ci = np.array([beta1 - z * se, beta1 + z * se])

    return t_coeff, se, t_ci[0], t_ci[1]


def plot_gate(est, x_test, y_test, t_test):
    """
    Plots the Group Average Treatment Effect (GATE) based on the model's effect estimates.

    Parameters:
    est (CausalForestDML): Trained Causal Forest DML model.
    x_test (pd.DataFrame): DataFrame containing the test predictors.
    y_test (pd.Series): Series containing the test outcomes.
    t_test (pd.Series): Series containing the test treatments.

    Returns:
    None: Saves the GATE plot as a PDF file.
    """
    effects = est.effect_inference(x_test).summary_frame()
    testing_frame = pd.DataFrame({'y': y_test, 't': t_test})
    testing_frame['effect'] = effects['point_estimate']
    testing_frame['treat_policy'] = np.where(testing_frame['effect'] < 0, 1, 0)
    print(np.sum(testing_frame['treat_policy']))
    print(np.sum(len(testing_frame)))

    # Calculate ATE with confidence intervals for groups
    ate_pi_1 = get_ate_w_ci(testing_frame[testing_frame.treat_policy == 1], y="y", t="t")
    ate_pi_0 = get_ate_w_ci(testing_frame[testing_frame.treat_policy == 0], y="y", t="t")

    # Plot GATE
    plt.figure(figsize=(8, 6))
    plt.bar(x=[0, 1], height=[ate_pi_0[0], ate_pi_1[0]], yerr=[abs(ate_pi_0[2] - ate_pi_0[0]), abs(ate_pi_1[2] - ate_pi_1[0])],
            capsize=10, color=["red", "green"], alpha=0.7)
    plt.plot([-0.5, 1.5], [0, 0], color="black")
    plt.xlim([-0.5, 1.5])
    plt.xticks([0, 1], ["$\pi(x_i)=0$\n\nCML model would not treat", "$\pi(x_i)=1$\n\nCML model would treat"], fontsize=18)
    plt.ylabel('Group ATE on returns', fontsize=18)
    plt.tight_layout()
    plt.savefig("plots/gate_plot.pdf")
    plt.close()
    print("GATE plot saved as plots/gate_plot.pdf")


def get_ipw_mean_w_se(dataset, prediction, y, q, asc):
    """
    Calculates the Inverse Propensity Weighted (IPW) estimate with standard errors
    relative to the random baseline, see Athey et al. (2023), arXiv:2310.08672.

    Parameters:
    dataset (pd.DataFrame): DataFrame containing the data.
    prediction (str): Column name for the prediction values.
    y (str): Column name for the outcome variable.
    q (float): Quantile for cutting the dataset (share treated).
    asc (bool): Whether to sort the dataset in ascending order.

    Returns:
    tuple: IPW estimate, lower bound, and upper bound of the confidence interval.
    """
    sum_t, data_n = sum(dataset.t), len(dataset)
    prop_score = sum_t / data_n
    dataset["f0"], dataset["f1"] = dataset[dataset.t == 0][y].mean(), dataset[dataset.t == 1][y].mean()

    # Sort data
    ordered_df = dataset.sort_values(prediction, ascending=asc).reset_index(drop=True).copy()

    # Get pi, ft
    cutoff = int(q * len(ordered_df))
    ordered_df["pi"] = 0
    ordered_df.iloc[:cutoff, ordered_df.columns.get_loc("pi")] = 1
    ordered_df["ft"] = np.where(ordered_df.t == 0, ordered_df["f0"], ordered_df["f1"])

    # Get IPW estimate
    policy_treat_nr = round(len(ordered_df) * q)
    treat_data = ordered_df.iloc[:policy_treat_nr, :]
    notrt_data = ordered_df.iloc[policy_treat_nr:, :]
    ipw_est_summand_1 = np.sum(treat_data[treat_data.t == 1][y]) / sum_t
    ipw_est_summand_2 = np.sum(notrt_data[notrt_data.t == 0][y]) / (data_n - sum_t)
    ipw_est_sum = ipw_est_summand_1 + ipw_est_summand_2

    # Get tau_hat_aipw
    tau_hat_aipw = ordered_df.prediction + (
            ((ordered_df.t - prop_score) / (prop_score * (1 - prop_score))) * (ordered_df[y] - ordered_df.ft))

    # Get delta_hat
    delta_hat = (1 / data_n) * np.sum((ordered_df.pi - q) * tau_hat_aipw)

    # Get standard errors
    se_hat = np.sqrt((1 / (data_n ** 2)) * np.sum(((ordered_df.pi - q) * tau_hat_aipw - delta_hat) ** 2))
    return ipw_est_sum, ipw_est_sum - 1.96 * se_hat, ipw_est_sum + 1.96 * se_hat


def cumulative_gain_ipw(dataset, prediction, y, asc=False):
    """
    Calculates cumulative gain using the IPW estimator.

    Parameters:
    dataset (pd.DataFrame): DataFrame containing the data.
    prediction (str): Column name for the predicted effects.
    y (str): Column name for the outcome variable.
    asc (bool): Whether to sort the dataset in ascending order.

    Returns:
    np.ndarray: Array containing the IPW estimates and their confidence intervals.
    """
    ipw_estimate_ci = np.zeros((101, 3))

    for q_100 in range(0, 101):
        q = q_100 / 100
        ipw_estimate_ci[q_100][0], ipw_estimate_ci[q_100][1], ipw_estimate_ci[q_100][2] = \
            get_ipw_mean_w_se(dataset, prediction, y, q, asc)

    return ipw_estimate_ci


def plot_cum_gain_ipw(x_test, y_test, t_test, est, asc=False):
    """
    Plots the cumulative gain using the IPW estimator.

    Parameters:
    x_test (pd.DataFrame): DataFrame containing the test predictors.
    y_test (pd.Series): Series containing the test outcomes.
    t_test (pd.Series): Series containing the test treatments.
    est (CausalForestDML): Trained Causal Forest DML model.
    asc (bool): Whether to sort the dataset in ascending order.

    Returns:
    None: Saves the plot as a PDF file.
    """
    plt.style.use(["grid", "science"])
    plt.figure(figsize=(8, 4.8))

    # Get predictions and prepare DataFrame
    nudge_pred = pd.DataFrame({"t": t_test, "y": y_test, "prediction": est.effect_inference(x_test).summary_frame().point_estimate.values})

    # Calculate cumulative gain
    cum_gain = cumulative_gain_ipw(nudge_pred, "prediction", "y", asc=asc)

    # Plot baseline and zero line
    plt.plot([0, 1], [cum_gain[0][0], cum_gain[100][0]], color="black", linestyle="--", label="Random Baseline")

    # Plot cumulative gain curve
    x = np.array(range(0, 101))
    plt.fill_between(x / 100, [share[1] for share in cum_gain], [share[2] for share in cum_gain], color="green", alpha=0.2)
    plt.plot(x / 100, [share[0] for share in cum_gain], color="green", alpha=1, label="Causal Forest")

    plt.xlabel("Share of customers treated")
    plt.ylabel("Mean returns (IPS estimator)")
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig("plots/ips_plot.pdf")
    plt.close()
    print("IPS estimator plot saved as plots/ips_plot.pdf")


def evaluate_model(test_data_path='data/test_data.csv', model_path='model/causal_forest_dml_model.pkl'):
    """
    Evaluates the trained Causal Forest DML model on the test data,
    and (i) plots both Group Average Treatment Effects (GATE)
    and (ii) the inverse propensity score estimator
    for different shares of treated individuals.

    Parameters:
    test_data_path (str): Path to the CSV file containing the test data.
    model_path (str): Path to the file containing the trained model.

    Returns:
    None: Saves the plots as .pdf in the folder "plots"
    """
    # Load the test dataset from the specified path
    test_df = pd.read_csv(test_data_path)

    # Split the data into predictors (X), treatment (T), and outcome (Y)
    y_test, t_test, x_test = test_df["Y"], test_df["T"], test_df.drop(columns=["T", "Y"])

    # Load the trained model from the specified path
    est = joblib.load(model_path)

    # (i) Group Average Treatment Effect (GATE) plot
    plot_gate(est, x_test, y_test, t_test)

    # (ii) IPS estimator plot for 0 - 1 shares treated
    plot_cum_gain_ipw(x_test, y_test, t_test, est, asc=True)


if __name__ == "__main__":
    # Call the evaluate_model function when the script is executed directly
    evaluate_model()
