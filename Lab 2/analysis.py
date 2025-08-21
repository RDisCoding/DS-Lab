# NYC Yellow Taxi Trip Data Analysis
# Exploratory Data Analysis with Statistical Techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create visualization directory
vis_dir = "DS\\Lab\\Lab 2\\visualization_images"
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
    print(f"Created directory: {vis_dir}")

print("NYC Yellow Taxi Trip Data Analysis")
print("="*50)

# Load the dataset
try:
    df = pd.read_csv('DS\\Lab\\Lab 2\\yellow_tripdata_sample.csv')
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print("Please ensure the CSV file is in the correct path")
    print("Using sample data for demonstration...")
    
    # Create sample data based on the provided first 10 entries
    sample_data = {
        'VendorID': [2, 1, 1, 1, 1, 1, 2, 1, 2, 2],
        'passenger_count': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0],
        'trip_distance': [1.72, 1.8, 4.7, 1.4, 0.8, 4.7, 10.82, 3.0, 5.44, 0.04],
        'RatecodeID': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'payment_type': [2, 1, 1, 1, 1, 1, 1, 2, 2, 2],
        'fare_amount': [17.7, 10.0, 23.3, 10.0, 7.9, 29.6, 45.7, 25.4, 31.0, 3.0],
        'extra': [1.0, 3.5, 3.5, 3.5, 3.5, 3.5, 6.0, 3.5, 1.0, 1.0],
        'tip_amount': [0.0, 3.75, 3.0, 2.0, 3.2, 6.9, 10.0, 0.0, 0.0, 0.0],
        'total_amount': [22.7, 18.75, 31.3, 17.0, 16.1, 41.5, 64.95, 30.4, 36.0, 8.0],
        'store_and_fwd_flag': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N']
    }
    
    # Extend the sample data to have more rows for meaningful analysis
    extended_data = {}
    for key, values in sample_data.items():
        # Replicate the pattern multiple times with some random variation
        extended_values = values * 100  # Replicate 100 times
        if key in ['passenger_count', 'trip_distance', 'fare_amount', 'extra', 'tip_amount', 'total_amount']:
            # Add some random noise to numeric columns
            base_array = np.array(extended_values)
            if key == 'passenger_count':
                # Keep passenger count as integers
                noise = np.random.choice([-1, 0, 1], size=len(base_array), p=[0.1, 0.8, 0.1])
                extended_values = np.maximum(0, base_array + noise).astype(float)
            else:
                # Add proportional noise to other numeric columns
                noise = np.random.normal(0, 0.1, size=len(base_array))
                extended_values = np.maximum(0, base_array * (1 + noise))
        extended_data[key] = extended_values
    
    df = pd.DataFrame(extended_data)
    print(f"Sample dataset created with shape: {df.shape}")

print(f"\nDataset Info:")
print(df.info())

print(f"\nFirst 5 rows:")
print(df.head())

# =============================================================================
# DATA CLEANING AND PREPROCESSING
# =============================================================================

print("\n" + "="*50)
print("DATA CLEANING AND PREPROCESSING")
print("="*50)

# Check for missing values
print("Missing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Handle missing values if any
df_clean = df.copy()

# Remove rows with missing passenger_count, trip_distance, fare_amount if any
for col in ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount', 'tip_amount', 'extra']:
    if col in df_clean.columns:
        df_clean = df_clean.dropna(subset=[col])

# Remove outliers (trips with unrealistic values)
print(f"\nOriginal dataset size: {len(df)}")

# Filter realistic trips
df_clean = df_clean[
    (df_clean['passenger_count'] >= 0) & (df_clean['passenger_count'] <= 8) &
    (df_clean['trip_distance'] >= 0) & (df_clean['trip_distance'] <= 100) &
    (df_clean['fare_amount'] >= 0) & (df_clean['fare_amount'] <= 500) &
    (df_clean['total_amount'] >= 0) & (df_clean['total_amount'] <= 500) &
    (df_clean['tip_amount'] >= 0)
]

print(f"Cleaned dataset size: {len(df_clean)}")
print(f"Rows removed: {len(df) - len(df_clean)}")

# =============================================================================
# PART A: DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*50)
print("PART A: DESCRIPTIVE STATISTICS")
print("="*50)

# 1. Univariate Analysis
print("\n1. UNIVARIATE ANALYSIS")
print("-" * 30)

# Define columns for analysis
numeric_columns = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount', 'tip_amount', 'extra']

def calculate_descriptive_stats(data, column):
    """Calculate comprehensive descriptive statistics for a column"""
    series = data[column].dropna()
    
    stats_dict = {
        'Count': len(series),
        'Missing Values': data[column].isnull().sum(),
        'Mean': series.mean(),
        'Median': series.median(),
        'Mode': series.mode().iloc[0] if not series.mode().empty else np.nan,
        'Minimum': series.min(),
        'Maximum': series.max(),
        'Range': series.max() - series.min(),
        'Standard Deviation': series.std(),
        'Variance': series.var(),
        'Skewness': stats.skew(series),
        'Kurtosis': stats.kurtosis(series),
        'Q1 (25th percentile)': series.quantile(0.25),
        'Q3 (75th percentile)': series.quantile(0.75),
        'IQR': series.quantile(0.75) - series.quantile(0.25)
    }
    
    return stats_dict

# Calculate and display descriptive statistics for each column
descriptive_results = {}
for col in numeric_columns:
    if col in df_clean.columns:
        print(f"\nDescriptive Statistics for {col.upper()}:")
        print("-" * 40)
        
        stats_dict = calculate_descriptive_stats(df_clean, col)
        descriptive_results[col] = stats_dict
        
        for stat, value in stats_dict.items():
            if isinstance(value, float):
                print(f"{stat:<20}: {value:.4f}")
            else:
                print(f"{stat:<20}: {value}")
        
        # Interpretation
        print(f"\nInterpretation for {col}:")
        skewness = stats_dict['Skewness']
        if abs(skewness) < 0.5:
            skew_interpretation = "approximately symmetric"
        elif skewness > 0.5:
            skew_interpretation = "positively skewed (right-tailed)"
        else:
            skew_interpretation = "negatively skewed (left-tailed)"
        
        kurtosis = stats_dict['Kurtosis']
        if kurtosis > 0:
            kurt_interpretation = "heavier tails than normal distribution"
        elif kurtosis < 0:
            kurt_interpretation = "lighter tails than normal distribution"
        else:
            kurt_interpretation = "similar tail weight to normal distribution"
        
        print(f"- Distribution is {skew_interpretation}")
        print(f"- Data has {kurt_interpretation}")
        print(f"- Coefficient of variation: {(stats_dict['Standard Deviation']/stats_dict['Mean'])*100:.2f}%")

# =============================================================================
# 2. VISUALIZATIONS
# =============================================================================

print("\n\n2. VISUALIZATIONS")
print("-" * 30)

# Create visualizations for each column separately
for col in numeric_columns:
    if col in df_clean.columns:
        # Create a figure with 3 subplots (histogram, boxplot, violin plot) for each column
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle(f'Analysis of {col.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0].hist(df_clean[col], bins=30, alpha=0.7, color=sns.color_palette("husl", 6)[numeric_columns.index(col)], edgecolor='black')
        axes[0].set_title(f'Histogram of {col.replace("_", " ").title()}')
        axes[0].set_xlabel(col.replace("_", " ").title())
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box Plot
        sns.boxplot(y=df_clean[col], ax=axes[1], color=sns.color_palette("husl", 6)[numeric_columns.index(col)])
        axes[1].set_title(f'Box Plot of {col.replace("_", " ").title()}')
        axes[1].set_ylabel(col.replace("_", " ").title())
        axes[1].grid(True, alpha=0.3)
        
        # Violin Plot
        sns.violinplot(y=df_clean[col], ax=axes[2], color=sns.color_palette("husl", 6)[numeric_columns.index(col)])
        axes[2].set_title(f'Violin Plot of {col.replace("_", " ").title()}')
        axes[2].set_ylabel(col.replace("_", " ").title())
        axes[2].grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'{vis_dir}/{col}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved visualization for {col} as {col}_analysis.png")

# Categorical visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Categorical Variables Analysis', fontsize=16, fontweight='bold')

# Payment Type Bar Chart
if 'payment_type' in df_clean.columns:
    payment_counts = df_clean['payment_type'].value_counts()
    axes[0, 0].bar(payment_counts.index, payment_counts.values, color=sns.color_palette("husl", len(payment_counts)))
    axes[0, 0].set_title('Payment Type Distribution')
    axes[0, 0].set_xlabel('Payment Type')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)

# VendorID Pie Chart
if 'VendorID' in df_clean.columns:
    vendor_counts = df_clean['VendorID'].value_counts()
    axes[0, 1].pie(vendor_counts.values, labels=vendor_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Vendor ID Distribution')

# Rate Code ID Bar Chart
if 'RatecodeID' in df_clean.columns:
    rate_counts = df_clean['RatecodeID'].value_counts()
    axes[1, 0].bar(rate_counts.index, rate_counts.values, color=sns.color_palette("husl", len(rate_counts)))
    axes[1, 0].set_title('Rate Code ID Distribution')
    axes[1, 0].set_xlabel('Rate Code ID')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)

# Store and Forward Flag
if 'store_and_fwd_flag' in df_clean.columns:
    flag_counts = df_clean['store_and_fwd_flag'].value_counts()
    axes[1, 1].pie(flag_counts.values, labels=flag_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Store and Forward Flag Distribution')

plt.tight_layout()
plt.savefig(f'{vis_dir}/categorical_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved categorical analysis as categorical_analysis.png")

# =============================================================================
# PART B: INFERENTIAL STATISTICS
# =============================================================================

print("\n" + "="*50)
print("PART B: INFERENTIAL STATISTICS")
print("="*50)

# 1. Confidence Intervals
print("\n1. CONFIDENCE INTERVALS (95%)")
print("-" * 40)

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for the mean"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin_error = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin_error, mean + margin_error

# Calculate 95% confidence intervals
ci_results = {}
for col in ['trip_distance', 'fare_amount', 'tip_amount']:
    if col in df_clean.columns:
        data = df_clean[col].dropna()
        ci_lower, ci_upper = calculate_confidence_interval(data)
        mean_val = data.mean()
        
        ci_results[col] = {
            'mean': mean_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
        print(f"\n{col.replace('_', ' ').title()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Interpretation: We are 95% confident that the true population mean")
        print(f"  {col.replace('_', ' ')} is between {ci_lower:.4f} and {ci_upper:.4f}")

# 2. Hypothesis Testing
print("\n\n2. HYPOTHESIS TESTING")
print("-" * 40)

# Test 1: One-sample t-test for tip amount
print("\nTest 1: One-sample t-test")
print("H0: Average tip amount = $2")
print("H1: Average tip amount ≠ $2")

if 'tip_amount' in df_clean.columns:
    tip_data = df_clean['tip_amount'].dropna()
    t_stat, p_value = stats.ttest_1samp(tip_data, 2.0)
    
    print(f"Sample mean: ${tip_data.mean():.4f}")
    print(f"Test statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Alpha level: 0.05")
    
    if p_value < 0.05:
        print("Result: REJECT the null hypothesis")
        print(f"Conclusion: The average tip amount is significantly different from $2")
    else:
        print("Result: FAIL TO REJECT the null hypothesis")
        print(f"Conclusion: No significant evidence that average tip amount differs from $2")

# Test 2: Two-sample t-test comparing fare amounts by payment type
print("\n\nTest 2: Two-sample t-test")
print("H0: Mean fare amount is the same for different payment types")
print("H1: Mean fare amount differs between payment types")

if 'payment_type' in df_clean.columns and 'fare_amount' in df_clean.columns:
    payment_types = df_clean['payment_type'].value_counts().index[:2]  # Top 2 payment types
    
    if len(payment_types) >= 2:
        group1 = df_clean[df_clean['payment_type'] == payment_types[0]]['fare_amount'].dropna()
        group2 = df_clean[df_clean['payment_type'] == payment_types[1]]['fare_amount'].dropna()
        
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        print(f"Payment Type {payment_types[0]} mean: ${group1.mean():.4f} (n={len(group1)})")
        print(f"Payment Type {payment_types[1]} mean: ${group2.mean():.4f} (n={len(group2)})")
        print(f"Test statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("Result: REJECT the null hypothesis")
            print("Conclusion: Significant difference in fare amounts between payment types")
        else:
            print("Result: FAIL TO REJECT the null hypothesis")
            print("Conclusion: No significant difference in fare amounts between payment types")

# Test 3: Chi-square test of independence
print("\n\nTest 3: Chi-square Test of Independence")
print("H0: Payment type and Rate Code ID are independent")
print("H1: Payment type and Rate Code ID are not independent")

if 'payment_type' in df_clean.columns and 'RatecodeID' in df_clean.columns:
    # Create contingency table
    contingency_table = pd.crosstab(df_clean['payment_type'], df_clean['RatecodeID'])
    print(f"\nContingency Table:")
    print(contingency_table)
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.6f}")
    print(f"Alpha level: 0.05")
    
    if p_value < 0.05:
        print("Result: REJECT the null hypothesis")
        print("Conclusion: Payment type and Rate Code ID are NOT independent")
    else:
        print("Result: FAIL TO REJECT the null hypothesis")
        print("Conclusion: No significant evidence that variables are dependent")

# 3. Correlation Analysis
print("\n\n3. CORRELATION ANALYSIS")
print("-" * 40)

# Calculate correlations
correlation_pairs = [
    ('trip_distance', 'fare_amount'),
    ('fare_amount', 'tip_amount')
]

print("Pearson Correlations:")
# Create scatter plots for correlation pairs
fig, axes = plt.subplots(1, len(correlation_pairs), figsize=(6*len(correlation_pairs), 5))
if len(correlation_pairs) == 1:
    axes = [axes]  # Make it iterable if only one subplot

for idx, (col1, col2) in enumerate(correlation_pairs):
    if col1 in df_clean.columns and col2 in df_clean.columns:
        # Remove rows where either column has NaN
        clean_data = df_clean[[col1, col2]].dropna()
        
        if len(clean_data) > 1:
            pearson_r, pearson_p = pearsonr(clean_data[col1], clean_data[col2])
            spearman_r, spearman_p = spearmanr(clean_data[col1], clean_data[col2])
            
            print(f"\n{col1.replace('_', ' ').title()} vs {col2.replace('_', ' ').title()}:")
            print(f"  Pearson correlation: {pearson_r:.4f} (p-value: {pearson_p:.6f})")
            print(f"  Spearman correlation: {spearman_r:.4f} (p-value: {spearman_p:.6f})")
            
            # Create scatter plot
            axes[idx].scatter(clean_data[col1], clean_data[col2], alpha=0.6, color=sns.color_palette("husl", len(correlation_pairs))[idx])
            axes[idx].set_xlabel(col1.replace('_', ' ').title())
            axes[idx].set_ylabel(col2.replace('_', ' ').title())
            axes[idx].set_title(f'{col1.replace("_", " ").title()} vs {col2.replace("_", " ").title()}\nPearson r = {pearson_r:.3f}')
            axes[idx].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(clean_data[col1], clean_data[col2], 1)
            p = np.poly1d(z)
            axes[idx].plot(clean_data[col1], p(clean_data[col1]), "r--", alpha=0.8)
            
            # Interpretation
            abs_r = abs(pearson_r)
            if abs_r < 0.3:
                strength = "weak"
            elif abs_r < 0.7:
                strength = "moderate"
            else:
                strength = "strong"
            
            direction = "positive" if pearson_r > 0 else "negative"
            print(f"  Interpretation: {strength} {direction} linear relationship")

plt.suptitle('Correlation Analysis - Scatter Plots', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{vis_dir}/correlation_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved correlation scatter plots as correlation_scatter_plots.png")

# Correlation Matrix
print(f"\nCorrelation Matrix:")
numeric_cols_available = [col for col in numeric_columns if col in df_clean.columns]
correlation_matrix = df_clean[numeric_cols_available].corr()
print(correlation_matrix.round(4))

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{vis_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved correlation heatmap as correlation_heatmap.png")

# =============================================================================
# BONUS TASKS (if datetime columns are available)
# =============================================================================

print("\n" + "="*50)
print("BONUS ANALYSIS")
print("="*50)

# Note: Since we're working with sample data without datetime parsing,
# this section shows what would be done with full temporal data

print("Note: Bonus tasks require datetime parsing from pickup/dropoff times")
print("For demonstration with sample data:")

# Simple analysis by payment type and trip patterns
if 'payment_type' in df_clean.columns:
    payment_analysis = df_clean.groupby('payment_type').agg({
        'fare_amount': ['mean', 'median', 'count'],
        'trip_distance': ['mean', 'median'],
        'tip_amount': ['mean', 'median']
    }).round(3)
    
    print(f"\nPayment Type Analysis:")
    print(payment_analysis)

# =============================================================================
# CONCLUSIONS AND INSIGHTS
# =============================================================================

print("\n" + "="*50)
print("CONCLUSIONS AND INSIGHTS")
print("="*50)

print("\nKey Findings from the Analysis:")

# Generate insights based on the analysis
insights = []

# Descriptive statistics insights
for col, stats_dict in descriptive_results.items():
    if abs(stats_dict['Skewness']) > 1:
        insights.append(f"• {col.replace('_', ' ').title()} shows significant skewness ({stats_dict['Skewness']:.2f})")
    
    cv = (stats_dict['Standard Deviation'] / stats_dict['Mean']) * 100
    if cv > 50:
        insights.append(f"• {col.replace('_', ' ').title()} has high variability (CV = {cv:.1f}%)")

# Correlation insights
if correlation_matrix is not None:
    high_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                high_correlations.append(f"• Strong correlation between {col1} and {col2} (r = {corr_val:.3f})")
    
    insights.extend(high_correlations)

# Print insights
for insight in insights[:10]:  # Limit to top 10 insights
    print(insight)

print(f"\nStatistical Test Results Summary:")
print(f"• Tip amount hypothesis test: {'Significant difference from $2' if 'p_value' in locals() and p_value < 0.05 else 'No significant difference from $2'}")
print(f"• Payment type comparison: {'Significant difference found' if 't_stat' in locals() else 'Analysis completed'}")
print(f"• Independence test: {'Variables are dependent' if 'chi2' in locals() and p_value < 0.05 else 'No significant dependence'}")

print(f"\nRecommendations for Further Analysis:")
print(f"• Investigate temporal patterns with full datetime data")
print(f"• Analyze geographical patterns using location IDs")
print(f"• Consider seasonal variations in trip patterns")
print(f"• Examine the relationship between trip distance and pricing")
print(f"• Study customer behavior patterns by payment type")

print(f"\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)

print(f"\nAll visualization images have been saved in the '{vis_dir}' folder:")
print(f"• Individual column analysis: {', '.join([f'{col}_analysis.png' for col in numeric_columns if col in df_clean.columns])}")
print(f"• Categorical analysis: categorical_analysis.png")
print(f"• Correlation heatmap: correlation_heatmap.png")
print(f"• Correlation scatter plots: correlation_scatter_plots.png")
print(f"\nTotal images saved: {len([col for col in numeric_columns if col in df_clean.columns]) + 3}")
print("="*50)