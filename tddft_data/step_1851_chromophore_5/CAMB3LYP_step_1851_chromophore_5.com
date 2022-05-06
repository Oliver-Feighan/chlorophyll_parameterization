%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1851_chromophore_5 TDDFT with cam-b3lyp functional

0 1
Mg   25.311   -7.388   47.424
C   27.763   -5.005   46.542
C   23.216   -6.203   45.027
C   23.316   -10.030   48.115
C   27.722   -9.020   49.407
N   25.495   -5.785   45.927
C   26.567   -4.879   45.766
C   26.260   -3.874   44.647
C   24.755   -4.218   44.286
C   24.432   -5.408   45.146
C   23.729   -3.029   44.573
C   27.192   -4.045   43.371
C   27.745   -2.730   42.747
C   27.732   -2.603   41.194
O   28.654   -2.397   40.475
O   26.400   -2.617   40.767
N   23.496   -8.033   46.678
C   22.825   -7.333   45.683
C   21.549   -8.091   45.402
C   21.512   -9.227   46.321
C   22.825   -9.153   47.103
C   20.642   -7.714   44.295
C   20.426   -10.297   46.295
O   19.444   -10.054   45.624
C   20.419   -11.575   47.192
N   25.552   -9.261   48.670
C   24.471   -10.003   48.883
C   24.802   -11.239   49.752
C   26.407   -11.168   49.901
C   26.551   -9.756   49.315
C   23.987   -11.301   51.072
C   27.312   -12.262   49.226
C   27.276   -12.282   47.660
N   27.239   -7.033   48.045
C   28.088   -7.852   48.720
C   29.321   -7.176   48.907
C   29.205   -6.050   48.186
C   27.996   -6.095   47.477
C   30.570   -7.644   49.640
C   29.874   -4.863   47.757
O   31.028   -4.548   47.964
C   28.916   -4.055   46.664
C   28.625   -2.675   47.050
O   28.108   -2.359   48.091
O   28.765   -1.856   45.985
C   28.424   -0.447   46.247
C   26.046   -2.185   39.457
C   25.921   -3.469   38.677
C   25.760   -3.672   37.356
C   25.640   -2.530   36.334
C   25.658   -5.052   36.719
C   24.322   -5.459   36.022
C   24.546   -5.627   34.529
C   23.710   -4.611   33.641
C   22.662   -5.160   32.697
C   24.735   -3.678   32.905
C   25.649   -4.222   31.807
C   25.907   -3.337   30.606
C   25.143   -3.478   29.253
C   23.899   -2.570   29.145
C   26.147   -3.227   28.094
C   25.750   -3.852   26.737
C   25.010   -2.874   25.805
C   23.526   -3.194   25.662
C   23.209   -3.889   24.259
C   22.654   -1.881   25.711
H   22.664   -5.736   44.208
H   22.538   -10.753   48.369
H   28.537   -9.489   49.962
H   26.435   -2.896   45.096
H   24.692   -4.662   43.293
H   23.243   -2.727   43.645
H   24.160   -2.209   45.148
H   22.919   -3.360   45.224
H   26.629   -4.370   42.496
H   28.024   -4.719   43.576
H   28.790   -2.578   43.019
H   27.250   -1.879   43.214
H   21.130   -8.176   43.437
H   20.657   -6.629   44.193
H   19.580   -7.959   44.313
H   20.470   -11.351   48.257
H   21.367   -12.052   46.941
H   19.504   -12.105   46.926
H   24.556   -12.110   49.145
H   26.616   -11.282   50.964
H   24.613   -11.773   51.829
H   23.091   -11.832   50.749
H   23.727   -10.296   51.404
H   26.756   -13.142   49.551
H   28.375   -12.224   49.461
H   28.197   -12.813   47.416
H   27.079   -11.287   47.261
H   26.382   -12.874   47.462
H   31.305   -8.004   48.920
H   30.416   -8.415   50.395
H   31.055   -6.915   50.289
H   29.461   -4.053   45.720
H   29.040   -0.056   47.057
H   27.417   -0.319   46.645
H   28.606   0.267   45.443
H   26.815   -1.502   39.096
H   25.131   -1.596   39.504
H   25.930   -4.373   39.286
H   25.780   -1.575   36.839
H   24.747   -2.556   35.709
H   26.410   -2.574   35.563
H   25.640   -5.741   37.564
H   26.566   -5.205   36.136
H   23.489   -4.807   36.284
H   24.103   -6.472   36.359
H   24.308   -6.607   34.115
H   25.625   -5.678   34.387
H   23.240   -3.938   34.359
H   22.505   -4.579   31.788
H   21.661   -5.000   33.098
H   22.695   -6.198   32.365
H   25.404   -3.230   33.640
H   24.181   -2.812   32.543
H   25.083   -5.042   31.366
H   26.502   -4.759   32.221
H   26.963   -3.361   30.336
H   25.753   -2.318   30.961
H   24.843   -4.523   29.183
H   24.239   -1.753   28.509
H   23.649   -2.153   30.120
H   23.098   -3.165   28.705
H   27.138   -3.618   28.324
H   26.251   -2.156   27.922
H   25.134   -4.714   26.990
H   26.600   -4.167   26.132
H   25.444   -2.906   24.806
H   25.093   -1.821   26.078
H   23.109   -3.943   26.336
H   22.153   -4.150   24.191
H   23.784   -4.768   23.967
H   23.432   -3.182   23.460
H   22.856   -1.511   26.716
H   21.575   -2.036   25.702
H   22.970   -1.107   25.011
