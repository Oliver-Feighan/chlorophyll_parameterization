%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1851_chromophore_9 TDDFT with PBE1PBE functional

0 1
Mg   36.650   1.252   30.020
C   34.066   2.370   32.195
C   38.708   1.189   32.667
C   38.978   0.570   27.852
C   34.293   1.613   27.318
N   36.379   1.803   32.175
C   35.256   2.222   32.899
C   35.496   2.411   34.377
C   37.000   2.030   34.481
C   37.440   1.614   33.034
C   37.908   3.131   35.025
C   34.572   1.504   35.270
C   33.883   2.064   36.526
C   34.825   2.984   37.367
O   34.824   4.199   37.426
O   35.859   2.195   37.877
N   38.568   0.951   30.269
C   39.289   0.992   31.481
C   40.589   0.521   31.224
C   40.751   0.405   29.815
C   39.403   0.588   29.216
C   41.627   0.342   32.229
C   42.049   0.147   28.872
O   41.995   0.052   27.623
C   43.300   -0.222   29.493
N   36.600   0.987   27.899
C   37.758   0.703   27.268
C   37.523   0.936   25.784
C   36.026   1.052   25.589
C   35.578   1.224   27.004
C   38.356   2.167   25.129
C   35.152   0.026   24.841
C   34.151   -0.880   25.613
N   34.588   1.940   29.702
C   33.779   2.034   28.561
C   32.473   2.403   28.941
C   32.547   2.523   30.350
C   33.865   2.250   30.760
C   31.398   2.943   28.135
C   31.758   2.962   31.515
O   30.661   3.481   31.520
C   32.695   2.817   32.728
C   32.824   4.098   33.456
O   33.579   5.041   33.168
O   31.951   4.111   34.519
C   31.896   5.348   35.349
C   36.948   2.914   38.533
C   37.047   2.342   39.944
C   37.636   2.872   41.010
C   38.422   4.133   40.908
C   37.520   2.255   42.402
C   38.770   1.506   42.888
C   39.664   2.269   43.889
C   41.126   1.746   43.999
C   41.091   0.209   44.546
C   41.917   2.663   44.955
C   43.397   2.322   44.953
C   44.249   3.410   45.609
C   44.942   4.277   44.507
C   46.459   4.200   44.423
C   44.476   5.778   44.646
C   44.479   6.585   43.297
C   45.298   7.885   43.330
C   44.571   9.218   43.516
C   44.162   9.745   42.087
C   45.435   10.205   44.337
H   39.311   0.942   33.543
H   39.737   0.381   27.091
H   33.652   1.848   26.465
H   35.310   3.459   34.612
H   37.207   1.222   35.184
H   38.618   3.585   34.334
H   38.548   2.922   35.882
H   37.263   3.972   35.281
H   35.151   0.668   35.665
H   33.872   1.067   34.559
H   33.608   1.186   37.110
H   32.950   2.577   36.290
H   42.442   0.999   31.924
H   41.993   -0.679   32.339
H   41.319   0.636   33.233
H   43.821   0.665   29.854
H   43.934   -0.719   28.760
H   43.235   -0.952   30.300
H   37.881   -0.006   25.368
H   35.809   2.022   25.142
H   38.881   2.662   25.945
H   37.592   2.839   24.738
H   39.004   1.853   24.311
H   35.806   -0.574   24.208
H   34.493   0.613   24.201
H   34.662   -1.242   26.505
H   34.025   -1.816   25.068
H   33.185   -0.456   25.885
H   31.915   3.616   27.451
H   30.569   3.351   28.714
H   31.115   2.046   27.584
H   32.218   2.089   33.384
H   32.690   6.079   35.197
H   31.809   5.059   36.396
H   30.956   5.899   35.340
H   36.728   3.967   38.710
H   37.837   2.634   37.967
H   36.342   1.544   40.176
H   38.749   4.205   39.871
H   39.281   3.903   41.538
H   37.861   5.003   41.251
H   36.620   1.659   42.554
H   37.362   3.063   43.115
H   39.500   1.419   42.083
H   38.522   0.523   43.290
H   39.089   2.061   44.791
H   39.574   3.293   43.527
H   41.466   1.655   42.967
H   41.802   0.130   45.368
H   41.417   -0.573   43.860
H   40.144   0.022   45.052
H   41.429   2.757   45.925
H   41.843   3.608   44.415
H   43.597   2.124   43.900
H   43.607   1.445   45.564
H   45.013   2.795   46.084
H   43.619   3.875   46.368
H   44.575   3.939   43.537
H   46.901   5.174   44.631
H   46.766   3.738   43.485
H   46.809   3.556   45.230
H   45.051   6.207   45.467
H   43.481   5.853   45.084
H   43.473   6.889   43.011
H   44.777   5.952   42.461
H   46.013   7.876   42.508
H   45.835   7.732   44.266
H   43.726   8.984   44.164
H   44.523   10.762   41.938
H   43.072   9.770   42.076
H   44.492   9.229   41.185
H   44.727   10.741   44.969
H   45.884   10.923   43.650
H   46.232   9.759   44.932
