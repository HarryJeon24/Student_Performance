import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# Load the data
train_df = pd.read_csv('original_data/train.csv')

train_missing_ratios = train_df.isna().sum() / len(train_df)

# Function to check if a column has unanimous values
def is_unanimous(column):
    return len(column.unique()) == 1

# Calculate unanimous columns
unanimous_columns = train_df.apply(is_unanimous)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(train_missing_ratios.index,
        train_missing_ratios.values,
        color=['red' if ratio == 1 else 'blue' if unanimous else 'orange' for ratio, unanimous in zip(train_missing_ratios.values, unanimous_columns)])
plt.xlabel('Feature', fontsize=12)
plt.ylabel('Missing values ratio', fontsize=12)
plt.title('Missing values in DATASET', fontsize=16)
plt.xticks(rotation=90)
plt.legend(handles=[mpatches.Patch(color='orange'),
                    mpatches.Patch(color='blue'),
                    mpatches.Patch(color='red')],
           labels=['Partially missing values', 'Unanimous values', 'Completely missing values'])

plt.tight_layout()
plt.show()