[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vaI1V9HX)
# fits 0

Run the notebook LeastSquareFit.ipynb to see the code that generated the images in the .pdf.  
  


This repository contains some background examples leading up to 
our discussion of fitting data.  

- CLT.ipynb: example of Central Limit Theorem
- LeastSquareFit[ROOT].ipynb: notebooks describing the exercise using numpy/matplotlib [or ROOT] tools
- LSQFit.C(py): starter code for the Least Squares Fitting exercise
- RandomMeasuresP1.C, RandomGaus.C, RandomMeasuresAndFitP1.C: code to generate movies linked to the class notes<br/>
eg. in root: <br>
root> .X RandomMeasuresP1.C

You will turn in an updated version of either LSQFit.C or PSQFit.py containing your work and the plots in the exercise desc


Other files:
- PlottingReview.ipynb a review of basic plot making in matplotlib and ROOT
- Interpolate.ipynb: Jupyter notebook illustrating the Lagrange interpolation and cubic splines.
- Lagrange.cpp: code to perform a Lagrange interpolation



STUDENT ANSWERS:

============================================================
SINGLE FIT EXAMPLE
============================================================
True parameters:   a = 0.500, b = 1.300, c = 0.500
Fitted parameters: a = 0.399, b = 1.265, c = 0.527
Chi-squared: 4.041
Reduced chi-squared: 0.449 (DOF = 9)

============================================================
MONTE CARLO STUDY
============================================================

Results from 1000 experiments:

Parameter a: mean = 0.4928 ± 0.1865 (true = 0.5)
Parameter b: mean = 1.3108 ± 0.2456 (true = 1.3)
Parameter c: mean = 0.4967 ± 0.0766 (true = 0.5)

Reduced chi-squared: mean = 0.991 ± 0.481
Expected mean for chi2_reduced = 1.0 (for DOF = 9)

All plots saved to 'LSQFit.pdf'

============================================================
OBSERVATIONS:
============================================================
1. The fitted parameters fluctuate around their true values
2. The correlations between parameters are visible in the 2D histograms
3. The reduced chi-squared distribution should center around 1.0

To study the effects:
- INCREASE npoints: uncertainties decrease, chi2 dist becomes narrower
- DECREASE npoints: uncertainties increase, less constrained fits
- INCREASE sigma: larger uncertainties, parameters vary more
- DECREASE sigma: tighter fits, chi2 may increase if model inadequate