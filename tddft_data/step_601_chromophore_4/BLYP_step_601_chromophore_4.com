%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_601_chromophore_4 TDDFT with blyp functional

0 1
Mg   9.135   2.975   28.092
C   10.269   1.548   31.090
C   7.826   5.436   30.025
C   7.411   3.798   25.486
C   10.179   0.030   26.410
N   8.817   3.262   30.317
C   9.499   2.647   31.364
C   9.461   3.447   32.698
C   8.442   4.589   32.316
C   8.394   4.408   30.767
C   7.012   4.439   32.962
C   10.818   4.002   33.165
C   10.751   4.696   34.593
C   11.502   3.977   35.801
O   12.579   3.363   35.744
O   10.773   4.057   36.980
N   7.976   4.515   27.728
C   7.534   5.465   28.601
C   6.832   6.533   27.858
C   6.649   6.013   26.568
C   7.341   4.708   26.547
C   6.500   7.861   28.504
C   5.890   6.682   25.393
O   5.864   6.186   24.290
C   5.268   8.133   25.584
N   8.973   2.112   26.108
C   7.982   2.607   25.253
C   7.763   1.658   24.015
C   8.993   0.749   24.145
C   9.369   0.966   25.596
C   6.375   0.951   24.171
C   10.194   1.054   23.239
C   10.332   0.163   21.956
N   10.186   1.258   28.562
C   10.560   0.179   27.778
C   11.244   -0.692   28.606
C   11.252   -0.191   29.874
C   10.565   1.007   29.819
C   11.959   -1.909   28.216
C   11.705   -0.487   31.197
O   12.277   -1.464   31.671
C   11.145   0.739   32.032
C   10.369   0.179   33.163
O   9.479   -0.694   33.093
O   10.749   0.772   34.334
C   10.019   0.381   35.554
C   11.360   3.382   38.167
C   11.164   4.189   39.407
C   11.374   3.823   40.678
C   12.033   2.495   41.033
C   11.150   4.736   41.862
C   9.761   5.262   42.147
C   9.640   6.701   41.577
C   9.346   7.716   42.692
C   7.990   7.436   43.337
C   9.339   9.232   42.100
C   10.719   9.918   42.315
C   10.680   11.419   42.152
C   11.281   12.048   43.393
C   11.750   13.501   43.068
C   10.380   12.105   44.642
C   10.701   11.212   45.856
C   9.933   9.840   45.815
C   9.316   9.540   47.190
C   10.246   9.196   48.406
C   8.119   8.504   47.137
H   7.275   6.178   30.607
H   6.682   4.098   24.730
H   10.394   -0.851   25.801
H   9.066   2.773   33.457
H   8.860   5.550   32.615
H   6.746   5.295   33.582
H   6.999   3.583   33.637
H   6.282   4.232   32.180
H   11.138   4.766   32.457
H   11.619   3.263   33.174
H   9.723   4.933   34.865
H   11.105   5.727   34.598
H   5.417   7.905   28.611
H   6.828   8.640   27.815
H   7.097   7.972   29.410
H   4.697   8.325   24.675
H   6.067   8.874   25.584
H   4.665   8.168   26.491
H   7.782   2.258   23.106
H   8.703   -0.282   23.941
H   5.961   1.045   23.167
H   5.655   1.518   24.762
H   6.482   -0.120   24.340
H   11.143   0.964   23.769
H   10.050   2.034   22.785
H   10.302   0.788   21.063
H   9.568   -0.609   21.864
H   11.289   -0.358   21.933
H   13.027   -1.843   28.422
H   11.820   -2.152   27.162
H   11.415   -2.692   28.744
H   11.976   1.285   32.478
H   9.617   1.175   36.183
H   10.833   -0.011   36.163
H   9.391   -0.506   35.466
H   12.434   3.274   38.018
H   10.860   2.425   38.316
H   10.554   5.067   39.194
H   11.423   1.871   41.687
H   12.933   2.685   41.618
H   12.491   1.997   40.179
H   11.730   5.632   41.643
H   11.523   4.306   42.792
H   9.557   5.177   43.214
H   9.028   4.657   41.612
H   8.793   6.667   40.891
H   10.462   6.992   40.924
H   10.082   7.625   43.490
H   7.400   8.348   43.432
H   8.059   6.885   44.275
H   7.359   6.762   42.759
H   8.503   9.745   42.576
H   9.193   9.210   41.020
H   11.414   9.629   41.527
H   11.146   9.638   43.279
H   9.627   11.693   42.080
H   11.126   11.749   41.214
H   12.238   11.584   43.635
H   12.422   13.722   42.239
H   12.363   13.817   43.913
H   10.902   14.184   43.112
H   9.425   11.776   44.235
H   10.310   13.143   44.967
H   10.488   11.616   46.846
H   11.756   10.940   45.894
H   10.664   9.064   45.585
H   9.114   9.878   45.096
H   8.753   10.413   47.519
H   10.582   10.098   48.918
H   11.175   8.764   48.033
H   9.917   8.430   49.108
H   8.604   7.687   46.603
H   7.237   8.969   46.698
H   7.705   8.129   48.073

