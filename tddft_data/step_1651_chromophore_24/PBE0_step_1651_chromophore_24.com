%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1651_chromophore_24 TDDFT with PBE1PBE functional

0 1
Mg   -0.285   43.855   24.361
C   1.870   43.456   27.114
C   -2.940   43.246   26.517
C   -2.282   44.338   21.840
C   2.582   43.835   22.283
N   -0.455   43.260   26.513
C   0.520   43.341   27.450
C   -0.092   43.120   28.894
C   -1.605   42.627   28.608
C   -1.684   43.141   27.147
C   -1.952   41.086   28.720
C   0.042   44.356   29.737
C   0.631   44.116   31.191
C   0.193   42.875   32.012
O   0.966   41.903   32.126
O   -1.004   42.961   32.666
N   -2.311   43.805   24.208
C   -3.234   43.637   25.192
C   -4.565   43.801   24.720
C   -4.411   44.061   23.289
C   -2.976   44.147   23.009
C   -5.765   43.720   25.604
C   -5.540   44.243   22.240
O   -5.241   44.526   21.050
C   -6.938   43.869   22.603
N   0.047   44.066   22.349
C   -0.953   44.412   21.531
C   -0.392   44.721   20.126
C   1.184   44.759   20.253
C   1.301   44.171   21.697
C   -0.894   43.804   19.019
C   1.818   46.134   20.263
C   0.969   47.341   20.829
N   1.787   43.698   24.536
C   2.793   43.627   23.618
C   4.116   43.570   24.285
C   3.749   43.578   25.671
C   2.340   43.536   25.770
C   5.459   43.582   23.618
C   4.332   43.452   26.921
O   5.492   43.349   27.317
C   3.108   43.316   27.879
C   3.225   42.045   28.605
O   3.410   40.931   28.126
O   3.331   42.326   29.913
C   3.844   41.343   30.881
C   -1.185   42.025   33.757
C   -2.656   41.730   33.995
C   -3.298   41.117   34.987
C   -2.569   40.398   36.118
C   -4.871   41.076   35.049
C   -5.471   42.194   35.919
C   -6.461   43.146   35.056
C   -7.940   43.164   35.594
C   -8.554   44.562   35.375
C   -8.865   42.124   34.904
C   -9.159   40.945   35.870
C   -10.357   41.288   36.813
C   -11.589   40.338   36.643
C   -12.820   41.134   36.233
C   -11.828   39.424   37.922
C   -12.798   38.261   37.790
C   -14.025   38.419   38.723
C   -15.187   37.469   38.604
C   -16.265   37.969   37.691
C   -15.795   37.221   40.029
H   -3.748   43.082   27.232
H   -2.876   44.713   21.004
H   3.469   44.020   21.673
H   0.541   42.276   29.166
H   -2.364   43.063   29.258
H   -1.064   40.459   28.803
H   -2.324   40.879   27.716
H   -2.721   40.788   29.432
H   -0.856   44.943   29.925
H   0.701   45.085   29.265
H   0.276   44.919   31.838
H   1.718   44.051   31.146
H   -6.178   42.724   25.445
H   -6.487   44.477   25.297
H   -5.586   43.851   26.671
H   -7.083   43.024   23.276
H   -7.404   43.651   21.642
H   -7.433   44.717   23.075
H   -0.760   45.696   19.808
H   1.686   44.068   19.576
H   -0.181   43.021   18.759
H   -1.309   44.385   18.196
H   -1.675   43.092   19.284
H   2.118   46.420   19.254
H   2.794   46.002   20.729
H   -0.026   47.190   21.248
H   0.870   47.940   19.924
H   1.513   47.849   21.625
H   5.381   43.241   22.585
H   6.085   42.816   24.075
H   5.863   44.587   23.737
H   3.065   44.172   28.552
H   4.285   41.868   31.728
H   4.640   40.810   30.362
H   3.114   40.630   31.266
H   -0.648   42.344   34.650
H   -0.709   41.091   33.459
H   -3.206   42.392   33.326
H   -2.668   41.050   36.985
H   -1.552   40.217   35.770
H   -2.993   39.394   36.115
H   -5.167   40.163   35.566
H   -5.125   41.029   33.990
H   -4.728   42.760   36.480
H   -6.185   41.693   36.573
H   -6.575   42.834   34.018
H   -6.037   44.150   35.054
H   -7.823   43.037   36.671
H   -7.766   45.302   35.511
H   -9.371   44.655   36.091
H   -8.906   44.659   34.348
H   -8.461   41.640   34.015
H   -9.869   42.425   34.606
H   -8.354   40.862   36.600
H   -9.033   39.970   35.399
H   -10.521   42.366   36.815
H   -10.013   41.053   37.821
H   -11.393   39.539   35.929
H   -12.647   42.184   35.995
H   -13.525   41.072   37.061
H   -13.303   40.741   35.338
H   -12.137   40.139   38.684
H   -10.881   38.955   38.191
H   -12.339   37.314   38.075
H   -13.095   38.139   36.749
H   -14.355   39.444   38.555
H   -13.545   38.298   39.694
H   -14.886   36.477   38.266
H   -15.975   37.734   36.667
H   -16.478   39.030   37.823
H   -17.180   37.446   37.969
H   -15.180   37.562   40.862
H   -15.958   36.147   40.122
H   -16.760   37.728   40.055

