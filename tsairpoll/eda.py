import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
preamble = r'''
            \usepackage{amsmath}
            \usepackage{libertine}
            \usepackage{xspace}

            '''
plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": "Computer Modern Roman",
    "text.latex.preamble": preamble,  'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
})


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


def perform_eda(df, features):
    """Perform exploratory data analysis on the dataset."""
    df = df.copy()
    df = df[features]
    print("Dataset Info:")
    print(df.info())

    # check if pm10 arpa is always positive
    if (df['PM10_ARPA'] < 0).any():
        print("Warning: PM10_ARPA contains negative values.")
    # check if pm10 is always positive
    if (df['PM10_MEAN'] < 0).any():
        print("Warning: PM10_MEAN contains negative values.")

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # Filter numerical columns for correlation analysis
    numeric_df = df.select_dtypes(include=["number"])

    # Plot heatmap for numerical feature correlations
    plt.figure(figsize=(15, 15))
    sns.heatmap(numeric_df.corr(method='pearson'), annot=True, cmap="coolwarm", fmt=".2f")
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

    # capitalize header and categorical values
    for col in categorical_columns:
        df.rename(columns={col: col.capitalize().replace("_", " ")}, inplace=True)
    categorical_columns = [col.capitalize().replace("_", " ") for col in categorical_columns]
    for col in categorical_columns:
        df[col] = df[col].str.capitalize().replace("_", " ")

    n_cats = len(categorical_columns)
    # make this 3x3 grid if more than 3
    if n_cats > 3:
        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), layout='constrained')
        axes = axes.flatten()
    else:
        fig, axes = plt.subplot_mosaic([[f"cat_{i}" for i in range(n_cats)]], figsize=(5 * n_cats, 5))

    for i, cat_col in enumerate(categorical_columns):
        ax = axes[i] if n_cats > 3 else axes[f"cat_{i}"]
        unique_categories = df[cat_col].dropna().unique()

        for category in unique_categories:
            subset = df[df[cat_col] == category]
            # cut plot at 0 for pm10 arpa
            sns.kdeplot(subset[target_column], label=category, fill=True, alpha=0.3, ax=ax, clip=(0, None))

        # Grid and ticks
        ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)

        ax.set_xlabel('ARPA PM10', fontsize=20)
        ax.set_ylabel("Density", fontsize=20)
        ax.set_ylim(0, 0.1)
        ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08])
        ax.set_xlim(-1, 125)
        ax.set_xticks([0, 25, 50, 75, 100, 120])
        if i in [0, 1, 2]:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        if i in [1, 2, 4, 5]:
            ax.set_ylabel('')
            ax.set_yticklabels([])
        ax.legend(title=cat_col, fontsize=12, title_fontsize=14)

        

    #plt.tight_layout()
    #plt.show()
    fig.savefig('Plots_eda_categorical_kde.pdf', dpi=800)

