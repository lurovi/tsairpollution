import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_three_lines(x, y1, y2, y3, labels, title="Line Plot"):
    plt.figure(figsize=(15, 15))
    plt.plot(x, y1, label=labels[0], linewidth=1.5, alpha=0.5, linestyle='-', markersize=4)
    plt.plot(x, y2, label=labels[1], linewidth=0.8, alpha=0.5, linestyle='--', markersize=4)
    plt.plot(x, y3, label=labels[2], linewidth=0.6, alpha=0.5, linestyle='-.', markersize=4)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def perform_eda(df):
    """Perform exploratory data analysis on the dataset."""

    print("Dataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Filter numerical columns for correlation analysis
    numeric_df = df.select_dtypes(include=["number"])

    # Plot heatmap for numerical feature correlations
    plt.figure(figsize=(15, 15))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # Plot distribution of each numerical variable
    num_cols = numeric_df.columns
    num_rows = (len(num_cols) + 2) // 3  # Arrange in a grid with 3 columns
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 8 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    # Plot histogram of the target variable
    target_column = df.columns[-1]  # Select last column as the target
    plt.figure(figsize=(8, 8))
    sns.histplot(df[target_column], bins=30, kde=True)
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.show()

    # Scatter plot between PM10 and the target variable
    plt.figure(figsize=(8, 8))
    plt.scatter(df.iloc[:, -2], df.iloc[:, -1], alpha=0.5)
    plt.xlabel(df.columns[-2])  # PM10
    plt.ylabel(df.columns[-1])  # Target variable
    plt.title(f"Scatter plot of {df.columns[-2]} vs {df.columns[-1]}")
    plt.show()

    # KDE plots for categorical variables w.r.t. target variable
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    n_cats = len(categorical_columns)
    fig, axes = plt.subplot_mosaic([[f"cat_{i}" for i in range(n_cats)]], figsize=(5 * n_cats, 5))

    for i, cat_col in enumerate(categorical_columns):
        ax = axes[f"cat_{i}"]
        unique_categories = df[cat_col].dropna().unique()

        for category in unique_categories:
            subset = df[df[cat_col] == category]
            sns.kdeplot(subset[target_column], label=category, fill=True, alpha=0.3, ax=ax)

        ax.set_title(f"KDE Plot of {target_column} by {cat_col}")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Density")
        ax.legend(title=cat_col)

    plt.tight_layout()
    plt.show()

