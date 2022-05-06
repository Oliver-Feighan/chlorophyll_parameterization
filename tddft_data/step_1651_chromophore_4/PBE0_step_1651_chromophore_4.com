%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1651_chromophore_4 TDDFT with PBE1PBE functional

0 1
Mg   9.306   3.516   28.318
C   10.947   1.944   30.980
C   7.611   5.385   30.786
C   7.585   4.696   25.920
C   10.297   0.744   26.328
N   9.154   3.547   30.589
C   10.052   2.933   31.457
C   9.767   3.433   32.956
C   8.445   4.228   32.842
C   8.372   4.402   31.324
C   7.119   3.504   33.300
C   11.074   4.134   33.505
C   11.531   3.917   34.940
C   10.572   3.344   35.922
O   10.336   2.168   36.054
O   10.041   4.350   36.643
N   7.905   4.870   28.379
C   7.424   5.685   29.427
C   6.546   6.685   28.801
C   6.612   6.544   27.329
C   7.403   5.311   27.155
C   5.806   7.806   29.656
C   6.057   7.454   26.303
O   6.143   7.236   25.098
C   5.374   8.707   26.678
N   8.954   2.841   26.417
C   8.214   3.495   25.561
C   7.932   2.715   24.220
C   8.915   1.565   24.391
C   9.481   1.715   25.760
C   6.408   2.371   23.993
C   9.906   1.309   23.259
C   11.310   2.050   23.381
N   10.520   1.820   28.473
C   10.807   0.752   27.614
C   11.816   -0.132   28.156
C   11.938   0.395   29.461
C   11.090   1.528   29.655
C   12.374   -1.394   27.620
C   12.595   0.090   30.686
O   13.322   -0.872   30.985
C   11.986   1.175   31.749
C   11.443   0.536   32.944
O   10.386   -0.022   32.956
O   12.428   0.582   33.927
C   12.117   -0.359   35.026
C   9.200   4.037   37.769
C   9.403   4.981   38.865
C   10.023   4.663   40.019
C   10.647   3.348   40.408
C   10.297   5.717   41.093
C   9.197   6.528   41.742
C   9.531   8.070   41.465
C   9.648   9.044   42.766
C   8.352   9.686   43.041
C   10.826   10.052   42.635
C   11.318   10.542   43.961
C   11.003   12.035   44.296
C   10.714   12.346   45.786
C   11.297   13.743   46.032
C   9.254   12.183   46.221
C   8.950   11.354   47.450
C   8.034   10.136   47.175
C   8.801   8.728   47.190
C   7.888   7.656   46.548
C   9.120   8.374   48.635
H   7.196   6.074   31.524
H   7.068   5.212   25.108
H   10.459   -0.058   25.605
H   9.574   2.568   33.590
H   8.536   5.266   33.162
H   7.336   2.519   33.713
H   6.490   3.321   32.428
H   6.420   4.130   33.854
H   10.906   5.211   33.487
H   12.021   3.986   32.987
H   11.794   4.874   35.391
H   12.345   3.192   34.909
H   4.758   7.960   29.399
H   6.385   8.719   29.517
H   5.906   7.575   30.716
H   6.062   9.199   27.365
H   4.458   8.372   27.165
H   5.059   9.251   25.788
H   8.161   3.277   23.314
H   8.284   0.678   24.449
H   6.367   1.319   23.713
H   6.034   2.990   23.178
H   5.810   2.531   24.891
H   9.546   1.642   22.286
H   10.146   0.246   23.222
H   12.190   1.426   23.534
H   11.283   2.596   24.324
H   11.529   2.756   22.580
H   11.828   -1.613   26.702
H   12.358   -2.186   28.369
H   13.449   -1.395   27.439
H   12.679   1.971   32.022
H   11.690   0.257   35.818
H   13.150   -0.616   35.261
H   11.488   -1.204   34.747
H   9.171   2.978   38.028
H   8.152   4.153   37.491
H   9.079   6.017   38.768
H   11.718   3.441   40.588
H   10.489   2.569   39.662
H   10.396   3.041   41.423
H   11.073   6.332   40.638
H   10.720   5.291   42.003
H   9.193   6.447   42.829
H   8.181   6.319   41.407
H   8.720   8.342   40.789
H   10.508   8.179   40.994
H   9.991   8.288   43.473
H   7.924   9.052   43.817
H   7.679   9.691   42.184
H   8.474   10.707   43.404
H   10.438   10.875   42.035
H   11.688   9.591   42.153
H   12.404   10.494   44.041
H   10.832   9.962   44.746
H   10.263   12.486   43.635
H   11.915   12.508   43.932
H   11.415   11.609   46.177
H   10.531   14.385   46.468
H   11.605   14.283   45.137
H   12.239   13.793   46.579
H   8.624   11.855   45.394
H   8.951   13.201   46.464
H   8.527   12.014   48.207
H   9.887   11.022   47.898
H   7.513   10.109   46.218
H   7.213   10.090   47.891
H   9.713   8.818   46.599
H   8.332   7.304   45.617
H   6.935   8.126   46.304
H   7.763   6.925   47.346
H   8.221   8.335   49.252
H   9.732   9.158   49.081
H   9.654   7.425   48.670
