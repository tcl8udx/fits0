import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import log
from random import gauss

# Configuration parameters
xmin = 1.0
xmax = 20.0
npoints = 12
sigma = 0.2
pars = [0.5, 1.3, 0.5]  # True parameters: a, b, c

def f(x, par):
    """The model function: f(x) = a + b*log(x) + c*log(x)^2"""
    return par[0] + par[1]*log(x) + par[2]*log(x)*log(x)

def getX(x):
    """Generate x values uniformly spaced"""
    step = (xmax - xmin) / npoints
    for i in range(npoints):
        x[i] = xmin + i * step

def getY(x, y, ey):
    """Generate y values with Gaussian noise"""
    for i in range(npoints):
        y[i] = f(x[i], pars) + gauss(0, sigma)
        ey[i] = sigma

def perform_fit(lx, ly, ley):
    """
    Perform least squares fit for f(x) = a + b*log(x) + c*log(x)^2
    Returns: fitted parameters [a, b, c] and chi-squared value
    """
    nPar = 3  # Three parameters: a, b, c
    nPnts = len(lx)
    
    # Create and fill the design matrix A
    A = np.matrix(np.zeros((nPnts, nPar)))
    for nr in range(nPnts):
        log_x = log(lx[nr])
        A[nr, 0] = 1.0           # Column for parameter a
        A[nr, 1] = log_x         # Column for parameter b
        A[nr, 2] = log_x * log_x # Column for parameter c
    
    # Apply weights (divide by uncertainties)
    for i in range(nPnts):
        A[i] = A[i] / ley[i]
    yw = (ly / ley).reshape(nPnts, 1)
    
    # Solve for parameters using normal equations
    # theta = (A^T A)^(-1) A^T y
    theta = inv(np.transpose(A).dot(A)).dot(np.transpose(A)).dot(yw)
    
    # Calculate chi-squared
    chi2 = 0.0
    for i in range(nPnts):
        y_fit = theta[0, 0] + theta[1, 0]*log(lx[i]) + theta[2, 0]*log(lx[i])*log(lx[i])
        chi2 += ((ly[i] - y_fit) / ley[i])**2
    
    return theta.A1, chi2  # Return as (flat) 1D array (as opposed to a 3x1 column vector) and chi2

# Open PDF file for saving all plots
pdf_filename = 'LSQFit.pdf'
pdf = PdfPages(pdf_filename)

# Example: single fit visualization
print("=" * 60)
print("SINGLE FIT EXAMPLE")
print("=" * 60)
lx = np.zeros(npoints)
ly = np.zeros(npoints)
ley = np.zeros(npoints)

getX(lx)
getY(lx, ly, ley)

fitted_pars, chi2 = perform_fit(lx, ly, ley)
dof = npoints - 3  # degrees of freedom
chi2_reduced = chi2 / dof

print(f"True parameters:   a = {pars[0]:.3f}, b = {pars[1]:.3f}, c = {pars[2]:.3f}")
print(f"Fitted parameters: a = {fitted_pars[0]:.3f}, b = {fitted_pars[1]:.3f}, c = {fitted_pars[2]:.3f}")
print(f"Chi-squared: {chi2:.3f}")
print(f"Reduced chi-squared: {chi2_reduced:.3f} (DOF = {dof})")

# Plot the single fit
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(lx, ly, yerr=ley, fmt='o', label='Data', capsize=5)

# Plot the fit curve
xi = np.linspace(xmin, xmax, 100)
yi = fitted_pars[0] + fitted_pars[1]*np.log(xi) + fitted_pars[2]*np.log(xi)**2
ax.plot(xi, yi, 'r-', linewidth=2, label='Fit')

# Also plot true function
yi_true = pars[0] + pars[1]*np.log(xi) + pars[2]*np.log(xi)**2
ax.plot(xi, yi_true, 'g--', linewidth=2, label='True function', alpha=0.7)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Single Pseudoexperiment\n$\\chi^2$/DOF = {chi2_reduced:.2f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig("justfit.png")
pdf.savefig(fig)  # Save to PDF

# Perform many experiments
print("\n" + "=" * 60)
print("MONTE CARLO STUDY")
print("=" * 60)
nexperiments = 1000
par_a = np.zeros(nexperiments)
par_b = np.zeros(nexperiments)
par_c = np.zeros(nexperiments)
chi2_set = np.zeros(nexperiments)
chi2_reduced = np.zeros(nexperiments)

for i in range(nexperiments):
    lx = np.zeros(npoints)
    ly = np.zeros(npoints)
    ley = np.zeros(npoints)
    
    getX(lx)
    getY(lx, ly, ley)
    
    fitted_pars, chi2 = perform_fit(lx, ly, ley)
    
    par_a[i] = fitted_pars[0]
    par_b[i] = fitted_pars[1]
    par_c[i] = fitted_pars[2]
    chi2_set[i] = chi2
    chi2_reduced[i] = chi2 / (npoints - 3)

# Print statistics
print(f"\nResults from {nexperiments} experiments:")
print(f"\nParameter a: mean = {np.mean(par_a):.4f} ± {np.std(par_a):.4f} (true = {pars[0]})")
print(f"Parameter b: mean = {np.mean(par_b):.4f} ± {np.std(par_b):.4f} (true = {pars[1]})")
print(f"Parameter c: mean = {np.mean(par_c):.4f} ± {np.std(par_c):.4f} (true = {pars[2]})")
print(f"\nReduced chi-squared: mean = {np.mean(chi2_reduced):.3f} ± {np.std(chi2_reduced):.3f}")
print(f"Expected mean for chi2_reduced = 1.0 (for DOF = {npoints - 3})")

# Create first 2x2 panel: Individual parameter distributions
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(pad=3.0)

# Plot 1: Parameter a distribution
axs1[0, 0].hist(par_a, bins=40, alpha=0.7, edgecolor='black', color='blue')
axs1[0, 0].axvline(pars[0], color='r', linestyle='--', linewidth=2, label=f'True ({pars[0]})')
axs1[0, 0].axvline(np.mean(par_a), color='cyan', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(par_a):.3f})')
axs1[0, 0].set_xlabel('Parameter a')
axs1[0, 0].set_ylabel('Frequency')
axs1[0, 0].set_title(f'Parameter a Distribution $\\mu$ = {np.mean(par_a):.4f}, $\\sigma$ = {np.std(par_a):.4f}')
axs1[0, 0].legend()
axs1[0, 0].grid(True, alpha=0.3)

# Plot 2: Parameter b distribution
axs1[0, 1].hist(par_b, bins=40, alpha=0.7, edgecolor='black', color='green')
axs1[0, 1].axvline(pars[1], color='r', linestyle='--', linewidth=2, label=f'True ({pars[1]})')
axs1[0, 1].axvline(np.mean(par_b), color='cyan', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(par_b):.3f})')
axs1[0, 1].set_xlabel('Parameter b')
axs1[0, 1].set_ylabel('Frequency')
axs1[0, 1].set_title(f'Parameter b Distribution $\\mu$ = {np.mean(par_b):.4f}, $\\sigma$ = {np.std(par_b):.4f}')
axs1[0, 1].legend()
axs1[0, 1].grid(True, alpha=0.3)

# Plot 3: Parameter c distribution
axs1[1, 0].hist(par_c, bins=40, alpha=0.7, edgecolor='black', color='orange')
axs1[1, 0].axvline(pars[2], color='r', linestyle='--', linewidth=2, label=f'True ({pars[2]})')
axs1[1, 0].axvline(np.mean(par_c), color='cyan', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(par_c):.3f})')
axs1[1, 0].set_xlabel('Parameter c')
axs1[1, 0].set_ylabel('Frequency')
axs1[1, 0].set_title(f'Parameter c Distribution $\\mu$ = {np.mean(par_c):.4f}, $\\sigma$ = {np.std(par_c):.4f}')
axs1[1, 0].legend()
axs1[1, 0].grid(True, alpha=0.3)

# Plot 4: Chi-squared distribution
axs1[1, 1].hist(chi2_set, bins=40, alpha=0.7, edgecolor='black', color='purple')
axs1[1, 1].set_xlabel(f'$\\chi^2$')
axs1[1, 1].set_ylabel('Frequency')
axs1[1, 1].set_title(f'$\\chi^2$ Distribution')
axs1[1, 1].grid(True, alpha=0.3)

#plt.savefig("LSQFit-First2x2.png")
pdf.savefig(fig1)  # Save to PDF


# Create second 2x2 panel: Parameter correlations
fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(pad=3.0)

# Plot 1: Parameter b vs a (2D histogram)
h1 = axs2[0, 0].hist2d(par_a, par_b, bins=30, cmap='Blues')
axs2[0, 0].axvline(pars[0], color='r', linestyle='--', linewidth=2, label='True a')
axs2[0, 0].axhline(pars[1], color='r', linestyle='--', linewidth=2, label='True b')
axs2[0, 0].set_xlabel('Parameter a')
axs2[0, 0].set_ylabel('Parameter b')
axs2[0, 0].set_title('Parameter b vs a')
axs2[0, 0].legend()
plt.colorbar(h1[3], ax=axs2[0, 0])

# Plot 2: Parameter c vs a (2D histogram)
h2 = axs2[0, 1].hist2d(par_a, par_c, bins=30, cmap='Greens')
axs2[0, 1].axvline(pars[0], color='r', linestyle='--', linewidth=2, label='True a')
axs2[0, 1].axhline(pars[2], color='r', linestyle='--', linewidth=2, label='True c')
axs2[0, 1].set_xlabel('Parameter a')
axs2[0, 1].set_ylabel('Parameter c')
axs2[0, 1].set_title('Parameter c vs a')
axs2[0, 1].legend()
plt.colorbar(h2[3], ax=axs2[0, 1])

# Plot 3: Parameter c vs b (2D histogram)
h3 = axs2[1, 0].hist2d(par_b, par_c, bins=30, cmap='Oranges')
axs2[1, 0].axvline(pars[1], color='r', linestyle='--', linewidth=2, label='True b')
axs2[1, 0].axhline(pars[2], color='r', linestyle='--', linewidth=2, label='True c')
axs2[1, 0].set_xlabel('Parameter b')
axs2[1, 0].set_ylabel('Parameter c')
axs2[1, 0].set_title('Parameter c vs b')
axs2[1, 0].legend()
plt.colorbar(h3[3], ax=axs2[1, 0])

# Plot 4: Reduced chi-squared distribution (repeated for completeness)
axs2[1, 1].hist(chi2_reduced, bins=40, alpha=0.7, edgecolor='black', color='purple')
axs2[1, 1].axvline(1.0, color='r', linestyle='--', linewidth=2, label='Expected (1.0)')
axs2[1, 1].axvline(np.mean(chi2_reduced), color='cyan', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(chi2_reduced):.2f})')
axs2[1, 1].set_xlabel(f'Reduced $\\chi^2$')
axs2[1, 1].set_ylabel('Frequency')
axs2[1, 1].set_title(f'Reduced $\\chi^2$ Distribution')
axs2[1, 1].legend()
axs2[1, 1].grid(True, alpha=0.3)

#plt.savefig("LSQFit-Second2x2.png")
pdf.savefig(fig2)  # Save to PDF
# Close the PDF file
pdf.close()
print(f"\nAll plots saved to '{pdf_filename}'")


print("\n" + "=" * 60)
print("OBSERVATIONS:")
print("=" * 60)
print("1. The fitted parameters fluctuate around their true values")
print("2. The correlations between parameters are visible in the 2D histograms")
print("3. The reduced chi-squared distribution should center around 1.0")
print("\nTo study the effects:")
print("- INCREASE npoints: uncertainties decrease, chi2 dist becomes narrower")
print("- DECREASE npoints: uncertainties increase, less constrained fits")
print("- INCREASE sigma: larger uncertainties, parameters vary more")
print("- DECREASE sigma: tighter fits, chi2 may increase if model inadequate")
