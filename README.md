# NeuralNetworkS2S
This repository contains files with Python code for the algorithms discussed in the paper 'Using Artificial Neural Networks for Generating Probabilistic Subseasonal Precipitation Forecasts over California', submitted to Monthly Weather Review. The following gives a brief description of the individual files:

- ANN-CalculateEnsembleStatistics.py: Reads the ensemble forecasts and calculates, separately for each member, the probability integral transform relative to the model climatology

- ANN-CalculateObsCategories.py: Reads the analysis data, calculates the climatology-dependent category boundaries, and uses them to categorize the analyzed precipitation amounts

- ANN-CalculateVerificationMetrics.py: Calculates various verification metrics for the ANN, CSGD, raw ensemble, and climatological probabilistic forecasts

- ANN-FindTuningParameters.py: Calculates cross-validated scores for different ANN architectures and selects the optimal regularization parameters

- ANN-GenerateProbabilityForecasts.py: Calculates probability forecasts based on the selected ANN model with optimal regularization parameters

- CNN-CalculateLargeScalePredictors.py: Reads ensemble forecast and analyzed Z500 and TCW data, upscales them to 1 degree and 7-day averages, and saves as .npz file

- CNN-CalculateVerificationMetrics.py: Calculates various verification metrics for the CNN probabilistic forecasts, based on either analyzed or forecast Z500/TCW fields

- CNN-FindTuningParameters.py: Calculates cross-validated scores for different CNN architectures and selects the optimal dropout rate for each of them

- CNN-FitConvolutionalNetworkModel.py: Fits a CNN model based on the selected CNN architecture with optimal dropout rate

- CNN-GenerateProbabilityForecasts.py: Estimates the adjustment factor and calculates adjusted probability forecasts based on the forecast Z500/TCW fields

- CodeForGraphics.py: Python code used to generate the figures in the MWR paper and a few additional figures used for presentations

- CSGD-FitClimatologicalDistributions.py: Fits climatological censored, shifted gamma distributions to the analyzed precipitation amounts

- CSGD-GenerateForecastDistributions.py: Fits a simplified CSGD model that links forecast and analyzed precipitation data, and generates probabilistic forecasts

- S-ANN-GenerateProbabilityForecasts.py: Calculates probability forecasts for the ANN model discussed in 'SupplementB.pdf'

- S-CalculateVerificationMetrics.py: Calculates various verification metrics for the additional experiments with ANN, CNN, and CSGD in the supplements

- S-CNN-FindTuningParameters.py: Calculates cross-validated scores for different CNN architectures discussed in 'SupplementC.pdf'

- S-CNN-FitConvolutionalNetworkModel.py: Fits a CNN model with optimal dropout rate for the additional experiments in 'SupplementC.pdf'

- S-CodeForGraphics.py: Python code used to generate the figures in 'SupplementA.pdf', 'SupplementB.pdf', and 'SupplementC.pdf'

- S-CSGD-GenerateForecastDistributions.py: Fits other variants of the CSGD model discussed in 'SupplementA.pdf' and generates probabilistic forecasts


