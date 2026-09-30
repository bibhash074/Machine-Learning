# ============================================================
# AUTOMATED ML EXPERIMENT
# 5 Algorithms × 3 Test Sizes × 5 PCA Components = 75 JPGs
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

# PCA values
PCA_COMPONENTS = [2, 4, 6, 8, 9]

# Test sizes
TEST_SIZES = [0.2, 0.4, 0.6]

# Output folder
OUTPUT_FOLDER = "75_ML_Results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD ALL THREE DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

df1 = pd.read_csv(
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    low_memory=True
)

df2 = pd.read_csv(
    "Wednesday-workingHours.pcap_ISCX.csv",
    low_memory=True
)

df3 = pd.read_csv(
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    low_memory=True
)

print("Tuesday rows   :", len(df1))
print("Wednesday rows :", len(df2))
print("Thursday rows  :", len(df3))


# ============================================================
# 3. COMBINE DATASETS
# ============================================================

dataset = pd.concat(
    [df1, df2, df3],
    ignore_index=True
)

print("\nCombined dataset shape:")
print(dataset.shape)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

dataset.columns = dataset.columns.str.strip()


# ============================================================
# 5. SEPARATE X AND Y
# ============================================================

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# ============================================================
# 6. CONVERT X TO NUMERIC
# ============================================================

print("\nConverting features to numeric...")

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 7. REPLACE INF WITH NaN
# ============================================================

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 8. IMPUTE MISSING VALUES
# ============================================================

print("Imputing missing values...")

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean"
)

X = imputer.fit_transform(X)


# ============================================================
# 9. ENCODE TARGET
# ============================================================

print("Encoding target labels...")

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

class_names = labelencoder_y.classes_

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "=", name)


# ============================================================
# 10. MAKE SURE DATA IS NUMERIC
# ============================================================

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y)


# ============================================================
# 11. INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

print("Total samples :", X.shape[0])
print("Total features:", X.shape[1])
print("Total classes :", len(class_names))

print("\nPCA components:", PCA_COMPONENTS)
print("Test sizes:", TEST_SIZES)

print(
    "\nTotal experiments:",
    len(PCA_COMPONENTS)
    * len(TEST_SIZES)
    * 5
)


# ============================================================
# 12. EXPERIMENT COUNTER
# ============================================================

experiment_number = 0

results = []


# ============================================================
# 13. MAIN AUTOMATION LOOP
# ============================================================

for test_size in TEST_SIZES:

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TEST SIZE =", test_size)
    print("=" * 70)

    X_train_original, X_test_original, y_train, y_test = train_test_split(

        X,
        y,

        test_size=test_size,

        random_state=0,

        stratify=y
    )


    # ========================================================
    # PCA LOOP
    # ========================================================

    for n_components in PCA_COMPONENTS:

        print("\n")
        print("-" * 70)
        print(
            "PCA COMPONENTS =",
            n_components
        )
        print("-" * 70)


        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        pca = PCA(
            n_components=n_components
        )

        # Fit PCA only on training data
        X_train = pca.fit_transform(
            X_train_original
        )

        # Transform test data
        X_test = pca.transform(
            X_test_original
        )


        # ====================================================
        # ALGORITHM 1
        # LINEAR REGRESSION
        # ====================================================

        algorithms = [

            (
                "Linear Regression",
                LinearRegression(),
                "regression"
            ),

            (
                "Logistic Regression",
                LogisticRegression(
                    random_state=0,
                    max_iter=1000
                ),
                "classification"
            ),

            (
                "Ridge Regression",
                Ridge(
                    alpha=1.0
                ),
                "regression"
            ),

            (
                "Lasso Regression",
                Lasso(
                    alpha=0.1
                ),
                "regression"
            ),

            (
                "Elastic Net",
                ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5
                ),
                "regression"
            )

        ]


        # ====================================================
        # ALGORITHM LOOP
        # ====================================================

        for algorithm_name, model, model_type in algorithms:

            experiment_number += 1

            print("\n")
            print("=" * 70)
            print(
                f"EXPERIMENT {experiment_number}/75"
            )
            print("Algorithm :", algorithm_name)
            print("Test size :", test_size)
            print("PCA       :", n_components)
            print("=" * 70)


            # ------------------------------------------------
            # TRAIN MODEL
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            y_pred_raw = model.predict(
                X_test
            )


            # =================================================
            # HANDLE REGRESSION MODELS
            # =================================================

            if model_type == "regression":

                # Regression gives continuous values.
                #
                # Example:
                # 2.73 -> 3
                # 4.12 -> 4
                #
                # We round to nearest class and make sure
                # prediction stays inside valid class range.

                y_pred = np.rint(
                    y_pred_raw
                ).astype(int)

                y_pred = np.clip(
                    y_pred,
                    0,
                    len(class_names) - 1
                )

            else:

                # Logistic Regression already gives
                # class labels.

                y_pred = y_pred_raw.astype(int)


            # =================================================
            # METRICS
            # =================================================

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=np.arange(len(class_names))
            )


            accuracy = accuracy_score(
                y_test,
                y_pred
            )


            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )


            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )


            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )


            # =================================================
            # CLASSIFICATION REPORT
            # =================================================

            report = classification_report(

                y_test,

                y_pred,

                labels=np.arange(
                    len(class_names)
                ),

                target_names=class_names,

                zero_division=0
            )


            # =================================================
            # PRINT RESULTS
            # =================================================

            print("\nConfusion Matrix:\n")
            print(cm)

            print(
                "\nAccuracy : {:.4f}".format(
                    accuracy
                )
            )

            print(
                "Precision: {:.4f}".format(
                    precision
                )
            )

            print(
                "Recall   : {:.4f}".format(
                    recall
                )
            )

            print(
                "F1 Score : {:.4f}".format(
                    f1
                )
            )

            print(
                "\nClassification Report:\n"
            )

            print(report)


            # =================================================
            # SAVE RESULTS TO LIST
            # =================================================

            results.append({

                "Experiment": experiment_number,

                "Algorithm": algorithm_name,

                "Test Size": test_size,

                "PCA Components": n_components,

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1 Score": f1

            })


            # =================================================
            # CREATE JPG
            # =================================================

            safe_algorithm_name = (
                algorithm_name
                .replace(" ", "_")
            )


            filename = (
                f"{experiment_number:02d}_"
                f"{safe_algorithm_name}_"
                f"Test_{int(test_size * 100)}_"
                f"PCA_{n_components}.jpg"
            )


            filepath = os.path.join(
                OUTPUT_FOLDER,
                filename
            )


            # =================================================
            # CREATE FIGURE
            # =================================================

            fig = plt.figure(
                figsize=(18, 13)
            )


            # =================================================
            # CONFUSION MATRIX
            # =================================================

            ax1 = plt.subplot2grid(
                (2, 2),
                (0, 0)
            )


            sns.heatmap(

                cm,

                annot=True,

                fmt="d",

                cmap="Blues",

                xticklabels=class_names,

                yticklabels=class_names,

                ax=ax1

            )


            ax1.set_title(
                "Confusion Matrix",
                fontsize=15,
                fontweight="bold"
            )


            ax1.set_xlabel(
                "Predicted Label"
            )


            ax1.set_ylabel(
                "Actual Label"
            )


            # =================================================
            # METRICS BOX
            # =================================================

            ax2 = plt.subplot2grid(
                (2, 2),
                (0, 1)
            )

            ax2.axis("off")


            metrics_text = f"""
ALGORITHM
{algorithm_name}

MODEL TYPE
{model_type}

TEST SIZE
{test_size}

PCA COMPONENTS
{n_components}

--------------------------------------

ACCURACY
{accuracy:.4f}

PRECISION
{precision:.4f}

RECALL
{recall:.4f}

F1 SCORE
{f1:.4f}
"""


            ax2.text(

                0.05,

                0.95,

                metrics_text,

                fontsize=13,

                verticalalignment="top",

                family="monospace"

            )


            # =================================================
            # CLASSIFICATION REPORT
            # =================================================

            ax3 = plt.subplot2grid(

                (2, 2),

                (1, 0),

                colspan=2

            )


            ax3.axis("off")


            ax3.text(

                0.01,

                0.98,

                "CLASSIFICATION REPORT\n\n"
                + report,

                fontsize=9,

                verticalalignment="top",

                family="monospace"

            )


            # =================================================
            # MAIN TITLE
            # =================================================

            fig.suptitle(

                f"{algorithm_name}   |   "
                f"Test Size = {test_size}   |   "
                f"PCA Components = {n_components}",

                fontsize=17,

                fontweight="bold"

            )


            # =================================================
            # SAVE JPG
            # =================================================

            plt.tight_layout(
                rect=[0, 0, 1, 0.95]
            )


            plt.savefig(

                filepath,

                dpi=150,

                format="jpg",

                bbox_inches="tight"

            )


            plt.close()


            print(
                "\nJPG saved:"
            )

            print(
                filepath
            )


# ============================================================
# 14. SAVE ALL 75 RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame(
    results
)


csv_path = os.path.join(
    OUTPUT_FOLDER,
    "ALL_75_RESULTS.csv"
)


results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n\n")
print("=" * 70)
print("              ALL EXPERIMENTS COMPLETED")
print("=" * 70)

print(
    "\nTotal experiments:",
    len(results)
)

print(
    "\nExpected JPG files: 75"
)

print(
    "Actual JPG files:",
    len([
        f
        for f in os.listdir(OUTPUT_FOLDER)
        if f.lower().endswith(".jpg")
    ])
)

print(
    "\nResults folder:"
)

print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)

print(
    "\nCSV file:"
)

print(
    os.path.abspath(
        csv_path
    )
)

print("\nDONE! 🎉")