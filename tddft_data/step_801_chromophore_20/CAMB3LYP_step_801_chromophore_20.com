%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_801_chromophore_20 TDDFT with cam-b3lyp functional

0 1
Mg   5.941   55.928   41.640
C   5.913   52.332   41.615
C   9.085   55.821   40.406
C   5.903   59.247   41.704
C   2.782   55.857   43.090
N   7.210   54.254   40.927
C   7.032   52.868   40.846
C   8.139   52.106   40.120
C   9.240   53.198   40.144
C   8.490   54.528   40.522
C   10.403   52.880   41.116
C   7.550   51.685   38.705
C   8.030   50.397   38.001
C   7.157   49.906   36.876
O   6.509   48.866   36.919
O   7.120   50.787   35.809
N   7.273   57.354   40.997
C   8.558   57.071   40.644
C   9.290   58.362   40.548
C   8.318   59.403   40.613
C   7.071   58.670   41.194
C   10.773   58.528   40.350
C   8.458   60.917   40.328
O   9.561   61.333   39.952
C   7.323   61.934   40.378
N   4.409   57.366   41.949
C   4.690   58.666   42.033
C   3.390   59.473   42.336
C   2.458   58.310   42.912
C   3.264   57.078   42.661
C   3.646   60.653   43.287
C   1.023   58.219   42.219
C   -0.151   58.137   43.182
N   4.451   54.447   42.081
C   3.310   54.565   42.844
C   2.884   53.289   43.244
C   3.831   52.355   42.762
C   4.827   53.093   42.076
C   1.665   52.969   44.089
C   4.246   50.921   42.723
O   3.688   49.964   43.272
C   5.614   50.907   42.009
C   5.467   49.875   40.970
O   4.647   49.911   40.067
O   6.354   48.866   41.201
C   6.189   47.779   40.228
C   6.341   50.240   34.694
C   6.897   51.068   33.518
C   6.669   50.832   32.241
C   5.561   49.988   31.679
C   7.274   51.777   31.166
C   6.460   53.097   31.226
C   6.451   53.709   29.804
C   6.933   55.162   29.934
C   5.986   56.183   29.221
C   8.329   55.340   29.357
C   9.133   56.362   30.219
C   10.414   55.807   30.884
C   11.614   55.954   30.026
C   12.635   54.839   30.124
C   12.242   57.354   30.288
C   12.776   58.190   29.125
C   14.198   58.532   29.218
C   15.112   57.327   28.858
C   16.345   57.121   29.814
C   15.758   57.494   27.404
H   10.147   55.815   40.149
H   6.045   60.327   41.783
H   1.808   55.895   43.582
H   8.473   51.306   40.781
H   9.697   53.267   39.158
H   10.438   51.803   41.278
H   10.238   53.501   41.996
H   11.353   53.207   40.691
H   7.770   52.553   38.083
H   6.466   51.601   38.629
H   8.220   49.652   38.773
H   8.978   50.633   37.518
H   11.153   59.303   41.015
H   10.944   59.026   39.395
H   11.421   57.682   40.578
H   7.256   62.266   41.414
H   6.355   61.484   40.157
H   7.502   62.765   39.695
H   2.989   59.919   41.426
H   2.482   58.499   43.985
H   2.747   60.805   43.884
H   3.836   61.518   42.651
H   4.533   60.584   43.915
H   1.073   57.361   41.548
H   0.798   59.100   41.618
H   -0.831   58.984   43.087
H   0.157   58.091   44.227
H   -0.658   57.202   42.945
H   0.887   52.604   43.419
H   1.108   53.759   44.594
H   1.890   52.114   44.727
H   6.284   50.461   42.744
H   6.944   47.986   39.470
H   5.254   47.746   39.670
H   6.407   46.763   40.556
H   5.294   50.400   34.952
H   6.409   49.165   34.527
H   7.516   51.924   33.788
H   6.050   49.349   30.944
H   4.927   50.624   31.061
H   4.917   49.424   32.354
H   7.455   51.221   30.247
H   8.271   51.931   31.579
H   6.869   53.629   32.086
H   5.436   52.920   31.554
H   5.414   53.740   29.469
H   7.043   53.089   29.131
H   7.017   55.428   30.987
H   5.457   56.681   30.034
H   5.108   55.780   28.716
H   6.409   56.943   28.565
H   8.468   55.731   28.349
H   8.903   54.427   29.197
H   8.481   56.921   30.890
H   9.477   57.152   29.551
H   10.232   54.741   31.022
H   10.676   56.164   31.880
H   11.324   56.000   28.977
H   12.886   54.443   29.140
H   12.270   54.040   30.769
H   13.555   55.253   30.536
H   12.809   57.262   31.214
H   11.440   58.011   30.624
H   12.170   59.030   28.785
H   12.649   57.572   28.236
H   14.413   58.804   30.252
H   14.351   59.349   28.514
H   14.502   56.424   28.824
H   17.189   56.832   29.187
H   16.047   56.235   30.374
H   16.566   57.960   30.474
H   15.701   56.576   26.819
H   16.794   57.831   27.426
H   15.126   58.211   26.878

