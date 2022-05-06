%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1501_chromophore_2 TDDFT with PBE1PBE functional

0 1
Mg   2.398   -0.093   44.412
C   5.947   0.358   43.984
C   1.919   2.557   42.263
C   -0.700   -1.046   43.936
C   3.171   -3.426   45.690
N   3.824   1.261   43.241
C   5.191   1.326   43.269
C   5.631   2.518   42.489
C   4.307   3.293   42.208
C   3.266   2.222   42.499
C   4.004   4.620   43.017
C   6.418   2.101   41.167
C   5.536   1.422   40.056
C   5.365   2.193   38.788
O   4.808   3.306   38.698
O   5.980   1.504   37.731
N   0.784   0.831   43.534
C   0.795   1.989   42.780
C   -0.660   2.344   42.590
C   -1.450   1.329   43.204
C   -0.454   0.289   43.593
C   -1.126   3.525   41.733
C   -2.881   1.391   43.397
O   -3.532   2.294   42.955
C   -3.619   0.182   43.858
N   1.355   -1.993   44.867
C   0.091   -2.151   44.342
C   -0.400   -3.604   44.429
C   0.739   -4.275   45.255
C   1.858   -3.158   45.332
C   -1.918   -3.814   44.924
C   1.259   -5.634   44.608
C   0.909   -6.902   45.426
N   4.117   -1.219   44.939
C   4.281   -2.474   45.527
C   5.712   -2.744   45.779
C   6.349   -1.627   45.194
C   5.378   -0.782   44.690
C   6.356   -3.954   46.346
C   7.651   -1.076   45.016
O   8.708   -1.557   45.455
C   7.457   0.127   44.054
C   8.318   1.274   44.462
O   8.959   1.941   43.659
O   8.316   1.405   45.801
C   9.278   2.367   46.338
C   5.725   2.035   36.418
C   5.907   0.837   35.463
C   5.289   -0.364   35.407
C   4.030   -0.626   36.180
C   5.670   -1.511   34.462
C   5.156   -1.445   33.048
C   5.290   -0.029   32.507
C   4.964   0.007   30.980
C   6.114   0.702   30.127
C   3.586   0.762   30.759
C   2.355   -0.208   30.812
C   1.438   0.083   31.968
C   0.429   1.230   31.706
C   -0.301   1.583   32.990
C   -0.630   0.760   30.715
C   -0.782   1.746   29.542
C   -2.025   2.727   29.834
C   -1.676   4.271   29.640
C   -2.323   5.143   30.693
C   -1.930   4.705   28.208
H   1.850   3.476   41.678
H   -1.670   -1.369   43.552
H   3.490   -4.385   46.104
H   6.251   3.184   43.090
H   4.439   3.615   41.175
H   3.067   4.501   43.561
H   3.911   5.282   42.156
H   4.827   4.765   43.717
H   7.192   1.395   41.467
H   6.787   3.050   40.777
H   4.565   1.221   40.508
H   6.010   0.488   39.754
H   -1.517   3.155   40.785
H   -0.241   4.109   41.483
H   -1.964   4.116   42.103
H   -3.261   -0.331   44.751
H   -3.648   -0.497   43.006
H   -4.646   0.468   44.085
H   -0.349   -4.018   43.422
H   0.493   -4.498   46.293
H   -2.680   -4.032   44.175
H   -2.315   -2.944   45.447
H   -2.001   -4.673   45.590
H   2.315   -5.719   44.349
H   0.872   -5.779   43.599
H   0.522   -6.597   46.398
H   1.776   -7.558   45.510
H   0.082   -7.450   44.973
H   6.152   -4.879   45.806
H   5.889   -4.133   47.315
H   7.433   -3.788   46.380
H   7.743   -0.290   43.089
H   8.998   2.222   47.382
H   9.032   3.379   46.017
H   10.323   2.107   46.169
H   6.319   2.875   36.057
H   4.712   2.406   36.261
H   6.833   0.954   34.901
H   4.285   -1.374   36.931
H   3.337   -0.983   35.417
H   3.642   0.291   36.623
H   5.222   -2.462   34.750
H   6.748   -1.655   34.396
H   4.089   -1.662   33.001
H   5.701   -2.115   32.383
H   6.273   0.376   32.748
H   4.634   0.632   33.073
H   4.911   -1.029   30.645
H   6.314   0.077   29.256
H   7.055   0.962   30.611
H   5.799   1.645   29.681
H   3.601   1.179   29.751
H   3.453   1.567   31.482
H   2.602   -1.269   30.859
H   1.748   -0.233   29.907
H   1.923   0.304   32.919
H   0.895   -0.854   32.095
H   1.061   2.088   31.476
H   0.346   2.089   33.706
H   -0.744   0.635   33.296
H   -1.177   2.210   32.825
H   -1.608   0.638   31.179
H   -0.513   -0.288   30.437
H   -0.951   1.117   28.668
H   0.155   2.274   29.367
H   -2.569   2.507   30.752
H   -2.784   2.469   29.095
H   -0.610   4.343   29.857
H   -3.099   5.708   30.177
H   -1.613   5.845   31.131
H   -2.686   4.552   31.534
H   -0.934   4.721   27.764
H   -2.184   5.765   28.235
H   -2.671   4.213   27.578

