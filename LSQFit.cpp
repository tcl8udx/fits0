#include "TRandom2.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TGClient.h"
#include "TStyle.h"
#include "TMatrixD.h"
#include "TMatrixDSym.h"
#include "TVectorD.h"
#include "TF1.h"
#include "TLine.h"
#include "TLegend.h"
#include "TPaveText.h"
#include <iostream>
#include <vector>
#include <cmath>

// Configuration parameters
const double xmin = 1.0;
const double xmax = 20.0;
const int npoints = 12;
const double sigma = 0.2;
const double pars[3] = {0.5, 1.3, 0.5}; // True parameters: a, b, c

// Random number generator
TRandom2 *rnd = new TRandom2();

// Model function: f(x) = a + b*log(x) + c*log(x)^2
double f(double x, const double par[3]) {
    double logx = TMath::Log(x);
    return par[0] + par[1]*logx + par[2]*logx*logx;
}

// Generate x values uniformly spaced
void getX(std::vector<double>& x) {
    double step = (xmax - xmin) / npoints;
    for (int i = 0; i < npoints; i++) {
        x[i] = xmin + i * step;
    }
}

// Generate y values with Gaussian noise
void getY(const std::vector<double>& x, std::vector<double>& y, std::vector<double>& ey) {
    for (int i = 0; i < npoints; i++) {
        y[i] = f(x[i], pars) + rnd->Gaus(0, sigma);
        ey[i] = sigma;
    }
}

// Perform least squares fit for f(x) = a + b*log(x) + c*log(x)^2
// Returns: fitted parameters and chi-squared value
void performFit(const std::vector<double>& lx, const std::vector<double>& ly, 
                const std::vector<double>& ley, double fitted_pars[3], double& chi2) {
    const int nPar = 3;  // Three parameters: a, b, c
    const int nPnts = lx.size();
    
    // Create and fill the design matrix A
    TMatrixD A(nPnts, nPar);
    TVectorD yw(nPnts);
    
    for (int nr = 0; nr < nPnts; nr++) {
        double log_x = TMath::Log(lx[nr]);
        A(nr, 0) = 1.0 / ley[nr];              // Column for parameter a (weighted)
        A(nr, 1) = log_x / ley[nr];            // Column for parameter b (weighted)
        A(nr, 2) = log_x * log_x / ley[nr];    // Column for parameter c (weighted)
        yw(nr) = ly[nr] / ley[nr];             // Weighted y values
    }
    
    // Solve for parameters using normal equations: theta = (A^T A)^(-1) A^T y
    TMatrixD AT(nPar, nPnts);
    AT.Transpose(A);
    TMatrixD ATA(nPar, nPar);
    ATA.Mult(AT, A);
    
    // Invert ATA
    TMatrixD ATA_inv = ATA.Invert();
    
    // Calculate AT * yw
    TVectorD ATy(nPar);
    ATy = AT * yw;
    
    // Calculate theta
    TVectorD theta(nPar);
    theta = ATA_inv * ATy;
    
    // Extract parameters
    fitted_pars[0] = theta(0);
    fitted_pars[1] = theta(1);
    fitted_pars[2] = theta(2);
    
    // Calculate chi-squared
    chi2 = 0.0;
    for (int i = 0; i < nPnts; i++) {
        double log_x = TMath::Log(lx[i]);
        double y_fit = fitted_pars[0] + fitted_pars[1]*log_x + fitted_pars[2]*log_x*log_x;
        double residual = (ly[i] - y_fit) / ley[i];
        chi2 += residual * residual;
    }
}

int main(int argc, char** argv) {
    TApplication theApp("App", &argc, argv);
    
    // Set random seed for reproducibility (comment out for different results each run)
    // rnd->SetSeed(12345);
    
    gStyle->SetOptStat(0);
    gStyle->SetPalette(kBird);
    
    std::cout << "============================================================" << std::endl;
    std::cout << "SINGLE FIT EXAMPLE" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Single fit example
    std::vector<double> lx(npoints), ly(npoints), ley(npoints);
    getX(lx);
    getY(lx, ly, ley);
    
    double fitted_pars[3];
    double chi2;
    performFit(lx, ly, ley, fitted_pars, chi2);
    
    int dof = npoints - 3;
    double chi2_reduced = chi2 / dof;
    
    std::cout << "True parameters:   a = " << pars[0] << ", b = " << pars[1] 
              << ", c = " << pars[2] << std::endl;
    std::cout << "Fitted parameters: a = " << fitted_pars[0] << ", b = " << fitted_pars[1] 
              << ", c = " << fitted_pars[2] << std::endl;
    std::cout << "Chi-squared: " << chi2 << std::endl;
    std::cout << "Reduced chi-squared: " << chi2_reduced << " (DOF = " << dof << ")" << std::endl;
    
    // Create canvas for single fit
    TCanvas *c1 = new TCanvas("c1", "Single Pseudoexperiment", 1000, 600);
    
    // Create TGraphErrors for data
    TGraphErrors *gr = new TGraphErrors(npoints);
    for (int i = 0; i < npoints; i++) {
        gr->SetPoint(i, lx[i], ly[i]);
        gr->SetPointError(i, 0, ley[i]);
    }
    gr->SetMarkerStyle(20);
    gr->SetMarkerSize(1.2);
    gr->SetTitle(Form("Single Pseudoexperiment (#chi^{2}/DOF = %.2f)", chi2_reduced));
    gr->GetXaxis()->SetTitle("x");
    gr->GetYaxis()->SetTitle("y");
    gr->Draw("AP");
    
    // Create fit function
    TF1 *fitFunc = new TF1("fitFunc", 
        [&](double *x, double *p) { 
            double logx = TMath::Log(x[0]);
            return fitted_pars[0] + fitted_pars[1]*logx + fitted_pars[2]*logx*logx;
        }, xmin, xmax, 0);
    fitFunc->SetLineColor(kRed);
    fitFunc->SetLineWidth(2);
    fitFunc->Draw("same");
    
    // Create true function
    TF1 *trueFunc = new TF1("trueFunc",
        [](double *x, double *p) {
            double logx = TMath::Log(x[0]);
            return pars[0] + pars[1]*logx + pars[2]*logx*logx;
        }, xmin, xmax, 0);
    trueFunc->SetLineColor(kGreen+2);
    trueFunc->SetLineWidth(2);
    trueFunc->SetLineStyle(2);
    trueFunc->Draw("same");
    
    TLegend *leg1 = new TLegend(0.15, 0.7, 0.4, 0.88);
    leg1->AddEntry(gr, "Data", "lep");
    leg1->AddEntry(fitFunc, "Fit", "l");
    leg1->AddEntry(trueFunc, "True function", "l");
    leg1->Draw();
    
    c1->Update();
    c1->SaveAs("LSQFit.pdf(");
    
    // Monte Carlo study
    std::cout << "\n============================================================" << std::endl;
    std::cout << "MONTE CARLO STUDY" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    const int nexperiments = 1000;
    std::vector<double> par_a(nexperiments), par_b(nexperiments), par_c(nexperiments);
    std::vector<double> chi2_set(nexperiments), chi2_reduced_vec(nexperiments);
    
    for (int i = 0; i < nexperiments; i++) {
        getX(lx);
        getY(lx, ly, ley);
        performFit(lx, ly, ley, fitted_pars, chi2);
        
        par_a[i] = fitted_pars[0];
        par_b[i] = fitted_pars[1];
        par_c[i] = fitted_pars[2];
        chi2_set[i] = chi2;
        chi2_reduced_vec[i] = chi2 / dof;
    }
    
    // Calculate statistics
    double mean_a = 0, mean_b = 0, mean_c = 0, mean_chi2_red = 0;
    for (int i = 0; i < nexperiments; i++) {
        mean_a += par_a[i];
        mean_b += par_b[i];
        mean_c += par_c[i];
        mean_chi2_red += chi2_reduced_vec[i];
    }
    mean_a /= nexperiments;
    mean_b /= nexperiments;
    mean_c /= nexperiments;
    mean_chi2_red /= nexperiments;
    
    double std_a = 0, std_b = 0, std_c = 0, std_chi2_red = 0;
    for (int i = 0; i < nexperiments; i++) {
        std_a += (par_a[i] - mean_a) * (par_a[i] - mean_a);
        std_b += (par_b[i] - mean_b) * (par_b[i] - mean_b);
        std_c += (par_c[i] - mean_c) * (par_c[i] - mean_c);
        std_chi2_red += (chi2_reduced_vec[i] - mean_chi2_red) * (chi2_reduced_vec[i] - mean_chi2_red);
    }
    std_a = TMath::Sqrt(std_a / nexperiments);
    std_b = TMath::Sqrt(std_b / nexperiments);
    std_c = TMath::Sqrt(std_c / nexperiments);
    std_chi2_red = TMath::Sqrt(std_chi2_red / nexperiments);
    
    std::cout << "\nResults from " << nexperiments << " experiments:" << std::endl;
    std::cout << "\nParameter a: mean = " << mean_a << " ± " << std_a << " (true = " << pars[0] << ")" << std::endl;
    std::cout << "Parameter b: mean = " << mean_b << " ± " << std_b << " (true = " << pars[1] << ")" << std::endl;
    std::cout << "Parameter c: mean = " << mean_c << " ± " << std_c << " (true = " << pars[2] << ")" << std::endl;
    std::cout << "\nReduced chi-squared: mean = " << mean_chi2_red << " ± " << std_chi2_red << std::endl;
    std::cout << "Expected mean for chi2_reduced = 1.0 (for DOF = " << dof << ")" << std::endl;
    
    // Create first 2x2 panel: Individual parameter distributions
    TCanvas *c2 = new TCanvas("c2", "Parameter Distributions", 1200, 1000);
    c2->Divide(2, 2);
    
    // Histogram 1: Parameter a
    c2->cd(1);
    TH1F *h_a = new TH1F("h_a", Form("Parameter a Distribution #mu = %.4f, #sigma = %.4f", mean_a, std_a),
                          40, mean_a - 4*std_a, mean_a + 4*std_a);
    for (int i = 0; i < nexperiments; i++) h_a->Fill(par_a[i]);
    h_a->SetFillColor(kBlue-9);
    h_a->SetLineColor(kBlack);
    h_a->GetXaxis()->SetTitle("Parameter a");
    h_a->GetYaxis()->SetTitle("Frequency");
    h_a->Draw();
    //TLine *line_a_true = new TLine(pars[0], 0, pars[0], h_a->GetMaximum());
    //line_a_true->SetLineColor(kRed);
    //line_a_true->SetLineStyle(2);
    //line_a_true->SetLineWidth(2);
    //line_a_true->Draw();
    //TLine *line_a_mean = new TLine(mean_a, 0, mean_a, h_a->GetMaximum());
    //line_a_mean->SetLineColor(kCyan+2);
    //line_a_mean->SetLineWidth(2);
    //line_a_mean->Draw();
    
    // Histogram 2: Parameter b
    c2->cd(2);
    TH1F *h_b = new TH1F("h_b", Form("Parameter b Distribution #mu = %.4f, #sigma = %.4f", mean_b, std_b),
                          40, mean_b - 4*std_b, mean_b + 4*std_b);
    for (int i = 0; i < nexperiments; i++) h_b->Fill(par_b[i]);
    h_b->SetFillColor(kGreen-9);
    h_b->SetLineColor(kBlack);
    h_b->GetXaxis()->SetTitle("Parameter b");
    h_b->GetYaxis()->SetTitle("Frequency");
    h_b->Draw();
    //TLine *line_b_true = new TLine(pars[1], 0, pars[1], h_b->GetMaximum());
    //line_b_true->SetLineColor(kRed);
    //line_b_true->SetLineStyle(2);
    //line_b_true->SetLineWidth(2);
    //line_b_true->Draw();
    //TLine *line_b_mean = new TLine(mean_b, 0, mean_b, h_b->GetMaximum());
    //line_b_mean->SetLineColor(kCyan+2);
    //line_b_mean->SetLineWidth(2);
    //line_b_mean->Draw();
    
    // Histogram 3: Parameter c
    c2->cd(3);
    TH1F *h_c = new TH1F("h_c", Form("Parameter c Distribution #mu = %.4f, #sigma = %.4f", mean_c, std_c),
                          40, mean_c - 4*std_c, mean_c + 4*std_c);
    for (int i = 0; i < nexperiments; i++) h_c->Fill(par_c[i]);
    h_c->SetFillColor(kOrange-9);
    h_c->SetLineColor(kBlack);
    h_c->GetXaxis()->SetTitle("Parameter c");
    h_c->GetYaxis()->SetTitle("Frequency");
    h_c->Draw();
    //TLine *line_c_true = new TLine(pars[2], 0, pars[2], h_c->GetMaximum());
    //line_c_true->SetLineColor(kRed);
    //line_c_true->SetLineStyle(2);
    //line_c_true->SetLineWidth(2);
    //line_c_true->Draw();
    //TLine *line_c_mean = new TLine(mean_c, 0, mean_c, h_c->GetMaximum());
    //line_c_mean->SetLineColor(kCyan+2);
    //line_c_mean->SetLineWidth(2);
    //line_c_mean->Draw();
    
    // Histogram 4: Chi-squared distribution
    c2->cd(4);
    double chi2_min = *std::min_element(chi2_set.begin(), chi2_set.end());
    double chi2_max = *std::max_element(chi2_set.begin(), chi2_set.end());
    TH1F *h_chi2 = new TH1F("h_chi2", "#chi^{2} Distribution", 40, chi2_min*0.8, chi2_max*1.2);
    for (int i = 0; i < nexperiments; i++) h_chi2->Fill(chi2_set[i]);
    h_chi2->SetFillColor(kViolet-9);
    h_chi2->SetLineColor(kBlack);
    h_chi2->GetXaxis()->SetTitle("#chi^{2}");
    h_chi2->GetYaxis()->SetTitle("Frequency");
    h_chi2->Draw();
    
    c2->Update();
    c2->SaveAs("LSQFit.pdf");
    
    // Create second 2x2 panel: Parameter correlations
    TCanvas *c3 = new TCanvas("c3", "Parameter Correlations", 1200, 1000);
    c3->Divide(2, 2);
    
    // 2D Histogram 1: b vs a
    c3->cd(1);
    TH2F *h2_ab = new TH2F("h2_ab", "Parameter b vs a", 30, mean_a - 3*std_a, mean_a + 3*std_a,
                            30, mean_b - 3*std_b, mean_b + 3*std_b);
    for (int i = 0; i < nexperiments; i++) h2_ab->Fill(par_a[i], par_b[i]);
    h2_ab->GetXaxis()->SetTitle("Parameter a");
    h2_ab->GetYaxis()->SetTitle("Parameter b");
    h2_ab->Draw("COLZ");
    //TLine *line_ab_a = new TLine(pars[0], mean_b - 3*std_b, pars[0], mean_b + 3*std_b);
    //line_ab_a->SetLineColor(kRed);
    //line_ab_a->SetLineStyle(2);
    //line_ab_a->SetLineWidth(2);
    //line_ab_a->Draw();
    //TLine *line_ab_b = new TLine(mean_a - 3*std_a, pars[1], mean_a + 3*std_a, pars[1]);
    //line_ab_b->SetLineColor(kRed);
    //line_ab_b->SetLineStyle(2);
    //line_ab_b->SetLineWidth(2);
    //line_ab_b->Draw();
    
    // 2D Histogram 2: c vs a
    c3->cd(2);
    TH2F *h2_ac = new TH2F("h2_ac", "Parameter c vs a", 30, mean_a - 3*std_a, mean_a + 3*std_a,
                            30, mean_c - 3*std_c, mean_c + 3*std_c);
    for (int i = 0; i < nexperiments; i++) h2_ac->Fill(par_a[i], par_c[i]);
    h2_ac->GetXaxis()->SetTitle("Parameter a");
    h2_ac->GetYaxis()->SetTitle("Parameter c");
    h2_ac->Draw("COLZ");
    //TLine *line_ac_a = new TLine(pars[0], mean_c - 3*std_c, pars[0], mean_c + 3*std_c);
    //line_ac_a->SetLineColor(kRed);
    //line_ac_a->SetLineStyle(2);
    //line_ac_a->SetLineWidth(2);
    //line_ac_a->Draw();
    //TLine *line_ac_c = new TLine(mean_a - 3*std_a, pars[2], mean_a + 3*std_a, pars[2]);
    //line_ac_c->SetLineColor(kRed);
    //line_ac_c->SetLineStyle(2);
    //line_ac_c->SetLineWidth(2);
    //line_ac_c->Draw();
    
    // 2D Histogram 3: c vs b
    c3->cd(3);
    TH2F *h2_bc = new TH2F("h2_bc", "Parameter c vs b", 30, mean_b - 3*std_b, mean_b + 3*std_b,
                            30, mean_c - 3*std_c, mean_c + 3*std_c);
    for (int i = 0; i < nexperiments; i++) h2_bc->Fill(par_b[i], par_c[i]);
    h2_bc->GetXaxis()->SetTitle("Parameter b");
    h2_bc->GetYaxis()->SetTitle("Parameter c");
    h2_bc->Draw("COLZ");
    //TLine *line_bc_b = new TLine(pars[1], mean_c - 3*std_c, pars[1], mean_c + 3*std_c);
    //line_bc_b->SetLineColor(kRed);
    //line_bc_b->SetLineStyle(2);
    //line_bc_b->SetLineWidth(2);
    //line_bc_b->Draw();
    //TLine *line_bc_c = new TLine(mean_b - 3*std_b, pars[2], mean_b + 3*std_b, pars[2]);
    //line_bc_c->SetLineColor(kRed);
    //line_bc_c->SetLineStyle(2);
    //line_bc_c->SetLineWidth(2);
    //line_bc_c->Draw();
    
    // Histogram 4: Reduced chi-squared
    c3->cd(4);
    TH1F *h_chi2_red = new TH1F("h_chi2_red", "Reduced #chi^{2} Distribution", 40, 0.2, 2.5);
    for (int i = 0; i < nexperiments; i++) h_chi2_red->Fill(chi2_reduced_vec[i]);
    h_chi2_red->SetFillColor(kViolet-9);
    h_chi2_red->SetLineColor(kBlack);
    h_chi2_red->GetXaxis()->SetTitle("Reduced #chi^{2}");
    h_chi2_red->GetYaxis()->SetTitle("Frequency");
    h_chi2_red->Draw();
    //TLine *line_expected = new TLine(1.0, 0, 1.0, h_chi2_red->GetMaximum());
    //line_expected->SetLineColor(kRed);
    //line_expected->SetLineStyle(2);
    //line_expected->SetLineWidth(2);
    //line_expected->Draw();
    //TLine *line_mean_chi2 = new TLine(mean_chi2_red, 0, mean_chi2_red, h_chi2_red->GetMaximum());
    //line_mean_chi2->SetLineColor(kCyan+2);
    //line_mean_chi2->SetLineWidth(2);
    //line_mean_chi2->Draw();
    
    c3->Update();
    c3->SaveAs("LSQFit.pdf)");
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "All plots saved to 'LSQFit.pdf'" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "OBSERVATIONS:" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "1. The fitted parameters fluctuate around their true values" << std::endl;
    std::cout << "2. The correlations between parameters are visible in the 2D histograms" << std::endl;
    std::cout << "3. The reduced chi-squared distribution should center around 1.0" << std::endl;
    std::cout << "\nTo study the effects:" << std::endl;
    std::cout << "- INCREASE npoints: uncertainties decrease, chi2 dist becomes narrower" << std::endl;
    std::cout << "- DECREASE npoints: uncertainties increase, less constrained fits" << std::endl;
    std::cout << "- INCREASE sigma: larger uncertainties, parameters vary more" << std::endl;
    std::cout << "- DECREASE sigma: tighter fits, chi2 may increase if model inadequate" << std::endl;
    
    theApp.Run();

}
