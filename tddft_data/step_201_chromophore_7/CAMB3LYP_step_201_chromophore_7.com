%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_201_chromophore_7 TDDFT with cam-b3lyp functional

0 1
Mg   26.286   -0.498   30.260
C   28.326   -0.511   32.957
C   23.534   0.000   32.345
C   24.362   0.011   27.533
C   28.966   -1.277   28.117
N   25.970   -0.487   32.463
C   27.001   -0.253   33.307
C   26.398   0.048   34.766
C   24.852   0.063   34.525
C   24.747   -0.216   33.030
C   23.966   -0.868   35.412
C   27.075   1.310   35.380
C   27.683   1.230   36.756
C   27.243   2.187   37.866
O   26.948   3.365   37.719
O   27.216   1.574   39.087
N   24.275   -0.059   30.039
C   23.296   0.027   30.948
C   22.102   0.373   30.402
C   22.226   0.365   28.984
C   23.669   0.061   28.780
C   20.762   0.423   31.194
C   21.239   0.656   27.872
O   21.473   0.660   26.660
C   19.755   0.925   28.181
N   26.635   -0.672   28.145
C   25.631   -0.349   27.253
C   26.162   -0.287   25.833
C   27.614   -0.860   26.021
C   27.806   -0.832   27.527
C   25.335   -1.013   24.759
C   28.690   0.009   25.248
C   29.690   -0.771   24.329
N   28.192   -0.969   30.414
C   29.221   -1.203   29.462
C   30.443   -1.396   30.161
C   30.159   -1.201   31.476
C   28.755   -0.886   31.593
C   31.756   -1.583   29.422
C   30.719   -1.211   32.802
O   31.876   -1.319   33.168
C   29.535   -0.716   33.845
C   29.354   -1.739   34.883
O   28.778   -2.827   34.734
O   29.975   -1.342   35.967
C   29.833   -2.219   37.124
C   26.887   2.325   40.261
C   25.447   2.226   40.742
C   24.867   2.658   41.848
C   25.619   3.370   42.992
C   23.401   2.377   42.073
C   23.220   1.262   43.186
C   22.272   1.585   44.343
C   23.057   1.840   45.671
C   23.126   0.555   46.600
C   22.443   2.981   46.564
C   23.439   4.243   46.741
C   22.874   5.638   46.252
C   22.398   6.611   47.438
C   20.988   6.255   47.927
C   22.618   8.097   47.131
C   21.681   8.708   46.083
C   22.546   9.208   44.831
C   21.545   9.779   43.806
C   22.028   9.302   42.481
C   21.425   11.313   43.970
H   22.739   0.149   33.079
H   23.704   0.270   26.701
H   29.814   -1.504   27.467
H   26.737   -0.790   35.376
H   24.471   1.083   34.566
H   22.898   -0.692   35.278
H   24.275   -0.645   36.433
H   24.207   -1.892   35.126
H   26.413   2.155   35.192
H   27.943   1.439   34.733
H   28.734   1.501   36.651
H   27.827   0.215   37.127
H   20.114   -0.379   30.841
H   20.345   1.417   31.028
H   20.885   0.241   32.262
H   19.696   1.808   28.817
H   19.508   0.073   28.815
H   19.233   1.047   27.232
H   26.219   0.750   25.500
H   27.670   -1.936   25.853
H   24.345   -1.220   25.165
H   25.717   -2.025   24.626
H   25.319   -0.457   23.822
H   29.325   0.449   26.017
H   28.383   0.814   24.580
H   30.514   -1.023   24.997
H   30.127   -0.174   23.530
H   29.306   -1.705   23.918
H   32.438   -1.309   30.226
H   31.839   -0.993   28.509
H   31.852   -2.647   29.207
H   29.853   0.253   34.232
H   28.968   -1.813   37.648
H   30.726   -2.121   37.741
H   29.799   -3.284   36.893
H   27.008   3.408   40.241
H   27.371   1.812   41.092
H   24.793   1.684   40.058
H   25.183   4.356   43.148
H   26.675   3.437   42.733
H   25.464   2.815   43.918
H   22.913   2.079   41.145
H   22.919   3.293   42.414
H   24.166   0.904   43.590
H   22.657   0.426   42.771
H   21.589   0.764   44.560
H   21.800   2.551   44.163
H   24.117   2.057   45.542
H   22.944   -0.365   46.045
H   22.492   0.586   47.486
H   24.095   0.506   47.097
H   22.001   2.635   47.498
H   21.557   3.299   46.013
H   24.357   3.968   46.221
H   23.702   4.386   47.789
H   22.017   5.416   45.616
H   23.544   6.279   45.679
H   23.003   6.439   48.328
H   20.248   7.056   47.909
H   21.074   5.954   48.971
H   20.432   5.477   47.403
H   23.667   8.171   46.847
H   22.690   8.641   48.073
H   21.224   9.523   46.645
H   20.995   7.933   45.742
H   23.055   8.363   44.367
H   23.279   9.931   45.190
H   20.550   9.401   44.044
H   23.117   9.245   42.472
H   21.704   9.875   41.612
H   21.644   8.297   42.311
H   22.090   11.739   44.721
H   20.406   11.663   44.131
H   21.772   11.763   43.039

