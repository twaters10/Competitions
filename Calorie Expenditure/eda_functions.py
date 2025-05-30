import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats

def univariate_analysis(df, var):
    """Comprehensive univariate analysis plots"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Univariate Analysis', fontsize=20, fontweight='bold')
    
    # Histogram with KDE
    sns.histplot(data=df, x=var, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution')
    
    # Box plot
    sns.boxplot(data=df, y=var, ax=axes[0, 1])
    axes[0, 1].set_title('Income Box Plot')
    
    # Violin plot
    sns.violinplot(data=df, y=var, ax=axes[0, 2])
    axes[0, 2].set_title('Satisfaction Violin Plot')
    
    # Density plot
    df[var].plot(kind='density', ax=axes[1, 2])
    axes[1, 2].set_title('Performance Score Density')
    
    # Q-Q plot for normality check
    stats.probplot(df[var].dropna(), dist="norm", plot=axes[2, 0])
    axes[2, 0].set_title('Performance Score Q-Q Plot')
    
    # Cumulative distribution
    df[var].plot(kind='hist', cumulative=True, density=True, ax=axes[2, 1], alpha=0.7)
    axes[2, 1].set_title('Salary Cumulative Distribution')
    
    # Strip plot with jitter
    sns.stripplot(data=df, y=var, ax=axes[2, 2], size=3, alpha=0.6)
    axes[2, 2].set_title('Bonus Strip Plot')
    
    plt.tight_layout()
    plt.show()