%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1051_chromophore_7 TDDFT with PBE1PBE functional

0 1
Mg   25.783   0.735   29.382
C   27.776   0.396   32.373
C   23.059   1.379   31.269
C   24.164   1.083   26.549
C   28.806   -0.275   27.621
N   25.478   0.717   31.631
C   26.380   0.706   32.606
C   25.718   0.850   33.921
C   24.181   0.945   33.530
C   24.192   1.020   32.029
C   23.403   -0.262   34.030
C   26.268   2.078   34.747
C   26.716   1.758   36.259
C   25.961   0.768   37.110
O   25.512   -0.291   36.680
O   25.548   1.322   38.314
N   23.801   1.027   28.907
C   22.876   1.436   29.875
C   21.729   1.786   29.112
C   21.984   1.754   27.743
C   23.346   1.256   27.694
C   20.362   1.935   29.829
C   21.000   1.919   26.561
O   21.351   1.739   25.385
C   19.601   2.221   26.830
N   26.320   0.186   27.416
C   25.429   0.543   26.408
C   26.194   0.418   25.058
C   27.690   0.207   25.387
C   27.618   -0.000   26.906
C   25.574   -0.650   24.115
C   28.548   1.371   24.835
C   29.981   1.015   24.353
N   27.931   0.169   29.843
C   28.958   -0.252   29.001
C   30.149   -0.424   29.780
C   29.734   -0.147   31.147
C   28.342   0.172   31.102
C   31.546   -0.888   29.261
C   30.109   -0.116   32.547
O   31.248   -0.363   32.940
C   28.946   0.341   33.425
C   28.721   -0.603   34.483
O   27.911   -1.482   34.521
O   29.613   -0.250   35.446
C   29.489   -1.070   36.627
C   24.475   0.777   39.088
C   24.370   1.402   40.554
C   25.375   1.541   41.402
C   26.745   0.958   41.298
C   25.275   2.501   42.517
C   24.648   1.789   43.652
C   23.904   2.736   44.523
C   24.059   2.489   46.069
C   23.840   0.985   46.410
C   22.960   3.292   46.839
C   23.312   3.511   48.351
C   23.205   4.969   48.850
C   24.616   5.672   48.686
C   25.287   5.714   50.064
C   24.630   7.082   48.044
C   24.760   7.046   46.504
C   23.955   8.244   45.879
C   22.524   7.818   45.525
C   22.326   7.658   44.001
C   21.591   8.866   46.052
H   22.214   1.658   31.902
H   23.773   1.568   25.652
H   29.612   -0.434   26.901
H   26.005   -0.073   34.424
H   23.660   1.742   34.060
H   23.180   -0.994   33.253
H   22.461   -0.066   34.542
H   23.977   -0.735   34.827
H   25.453   2.788   34.890
H   27.000   2.612   34.141
H   26.640   2.744   36.718
H   27.758   1.448   36.338
H   19.727   1.124   29.472
H   19.797   2.810   29.506
H   20.319   1.917   30.918
H   19.294   1.299   27.324
H   19.104   2.133   25.864
H   19.436   3.227   27.215
H   26.035   1.395   24.601
H   28.178   -0.691   25.008
H   24.697   -1.082   24.597
H   26.356   -1.406   24.048
H   25.357   -0.155   23.169
H   28.617   2.096   25.646
H   27.973   1.942   24.106
H   30.427   0.292   25.036
H   30.659   1.850   24.528
H   30.170   0.550   23.386
H   32.289   -0.427   29.912
H   31.889   -0.695   28.244
H   31.591   -1.964   29.433
H   29.089   1.290   33.941
H   28.512   -0.737   36.979
H   30.290   -0.704   37.269
H   29.638   -2.111   36.338
H   24.508   -0.310   39.162
H   23.616   0.981   38.448
H   23.513   2.006   40.853
H   27.526   1.677   41.546
H   27.044   0.500   40.355
H   26.859   0.148   42.019
H   24.931   3.498   42.241
H   26.261   2.770   42.896
H   25.455   1.408   44.278
H   24.135   0.889   43.315
H   22.842   2.816   44.292
H   24.349   3.708   44.309
H   25.049   2.778   46.422
H   23.139   0.769   47.216
H   24.823   0.652   46.745
H   23.484   0.293   45.647
H   21.968   2.875   46.664
H   23.099   4.259   46.355
H   24.274   3.053   48.576
H   22.609   2.951   48.968
H   22.705   4.853   49.812
H   22.581   5.596   48.213
H   25.323   5.018   48.175
H   25.616   4.750   50.452
H   24.676   6.195   50.827
H   26.252   6.220   50.031
H   25.499   7.677   48.323
H   23.699   7.568   48.338
H   24.309   6.117   46.155
H   25.770   6.993   46.097
H   24.503   8.474   44.965
H   24.055   9.120   46.520
H   22.325   6.862   46.008
H   21.397   8.050   43.587
H   22.419   6.625   43.666
H   23.097   8.178   43.433
H   22.039   9.652   46.661
H   20.771   8.419   46.613
H   21.134   9.408   45.224

