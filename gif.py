import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
import imageio
import os

# Step 1: Generate Bernoulli trial data (0.5 success probability)
np.random.seed(42)
num_trials = 20
data = np.random.binomial(1, 0.5, num_trials)

# Step 2: Create a DataFrame to store results
df = pd.DataFrame({
    'Trial': np.arange(1, num_trials + 1),
    'Result': data
})

# Calculate cumulative successes and failures for each trial
df['Cumulative_Successes'] = df['Result'].cumsum()
df['Cumulative_Failures'] = (df['Trial'] - df['Cumulative_Successes'])

# Step 3: Bayesian updating function
def calculate_posterior(successes, failures, alpha_prior=1, beta_prior=1):
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    return alpha_post, beta_post

# Directory to save images
output_dir = 'beta_distributions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store file paths for GIF creation
filenames = []

# Step 4: Precompute the maximum y-axis value across all trials
p_values = np.linspace(0, 1, 100)
max_density = 0  # Variable to track the global maximum density

# First loop to calculate the maximum density across all trials
for _, row in df.iterrows():
    successes = row['Cumulative_Successes']
    failures = row['Cumulative_Failures']
    
    # Calculate posterior distribution parameters
    alpha_post, beta_post = calculate_posterior(successes, failures)
    
    # Calculate the PDF of the Beta distribution
    beta_pdf = beta.pdf(p_values, alpha_post, beta_post)
    
    # Update the maximum density
    max_density = max(max_density, beta_pdf.max())

# Step 5: Plot posterior distributions and save each step as an image
# Use the precomputed max_density for consistent y-axis limits
for i, row in df.iterrows():
    plt.figure(figsize=(10, 8))
    
    # Plot the non-informative prior (Beta(1, 1)) first
    beta_pdf_prior = beta.pdf(p_values, 1, 1)
    plt.plot(p_values, beta_pdf_prior, label=f'Prior: Beta(1, 1)', color='gray', lw=2, linestyle='--')

    successes = row['Cumulative_Successes']
    failures = row['Cumulative_Failures']
    
    # Update posterior distribution
    alpha_post, beta_post = calculate_posterior(successes, failures)
    
    # Calculate Beta distribution for the posterior
    beta_pdf = beta.pdf(p_values, alpha_post, beta_post)
    
    # Plot the posterior distribution
    plt.plot(p_values, beta_pdf, label=f'Trial {int(row["Trial"])}: Beta({alpha_post}, {beta_post})', lw=2, color='black')
    
    # Plot a vertical line at the final expected value
    expected_value = alpha_post / (alpha_post + beta_post)
    plt.axvline(expected_value, color='red', linestyle='--', lw=2, label=f'Expected Value: {expected_value:.3f}')
    
    # Set axis limits based on precomputed maximum density
    plt.xlim(0, 1)
    plt.ylim(0, max_density * 1.1)  # Set y-axis slightly above the global max density
    
    # Add labels and title
    plt.title(f'Posterior Distributions After Each Trial (Beta Updates)\nTrial {int(row["Trial"])}', fontsize=16)
    plt.xlabel('Success Probability (p)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Show legend and grid
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    
    # Save the plot to a file
    filename = f"{output_dir}/trial_{i+1}.png"
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

# Step 6: Create GIF from the saved images with a delay
with imageio.get_writer('beta_distribution_updates.gif', mode='I', duration=1.0) as writer:  # 1.0 sec delay
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Optionally, remove image files after creating GIF
for filename in filenames:
    os.remove(filename)