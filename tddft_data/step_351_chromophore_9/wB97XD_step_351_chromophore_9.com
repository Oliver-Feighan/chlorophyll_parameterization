%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_351_chromophore_9 TDDFT with wB97XD functional

0 1
Mg   35.006   1.776   29.461
C   32.722   2.995   31.797
C   37.442   1.984   31.779
C   37.139   0.750   27.070
C   32.418   2.069   26.936
N   35.032   2.279   31.588
C   33.947   2.681   32.371
C   34.308   2.680   33.887
C   35.875   2.636   33.760
C   36.152   2.225   32.349
C   36.660   3.820   34.357
C   33.616   1.443   34.641
C   34.123   1.149   36.068
C   34.648   2.314   36.821
O   34.031   3.328   37.084
O   35.840   1.994   37.365
N   37.004   1.342   29.492
C   37.841   1.550   30.460
C   39.180   0.895   30.088
C   39.045   0.401   28.788
C   37.675   0.863   28.402
C   40.218   0.827   31.179
C   40.032   -0.455   28.000
O   39.817   -0.894   26.883
C   41.409   -0.771   28.559
N   34.774   1.384   27.343
C   35.859   0.913   26.555
C   35.568   0.933   25.063
C   33.978   1.179   25.087
C   33.672   1.523   26.536
C   36.408   1.953   24.272
C   33.167   -0.062   24.660
C   33.150   -1.353   25.517
N   33.029   2.420   29.398
C   32.131   2.386   28.337
C   30.823   2.839   28.793
C   31.044   3.137   30.151
C   32.383   2.856   30.455
C   29.543   2.981   27.978
C   30.516   3.670   31.343
O   29.472   4.202   31.544
C   31.472   3.326   32.552
C   31.529   4.435   33.551
O   32.139   5.453   33.307
O   31.058   4.030   34.715
C   31.331   4.759   35.951
C   36.461   3.032   38.210
C   37.490   2.342   39.027
C   37.568   2.070   40.353
C   36.485   2.492   41.426
C   38.615   1.111   40.880
C   39.836   1.793   41.555
C   40.044   1.352   43.006
C   41.216   0.450   43.258
C   40.848   -0.806   44.079
C   42.414   1.222   44.006
C   43.708   1.015   43.269
C   44.747   1.923   43.884
C   45.290   3.002   42.953
C   46.849   2.791   42.629
C   44.998   4.360   43.535
C   45.148   5.398   42.326
C   44.375   6.706   42.625
C   45.242   7.948   42.192
C   46.162   8.613   43.256
C   44.332   9.038   41.470
H   38.258   2.315   32.424
H   37.875   0.413   26.337
H   31.725   2.203   26.102
H   33.951   3.591   34.369
H   36.244   1.768   34.306
H   36.049   4.210   35.171
H   36.641   4.551   33.548
H   37.663   3.554   34.691
H   33.996   0.626   34.027
H   32.542   1.303   34.520
H   34.918   0.403   36.043
H   33.343   0.714   36.693
H   40.664   -0.167   31.152
H   39.935   1.254   32.141
H   41.050   1.483   30.922
H   41.716   0.098   29.141
H   42.102   -1.074   27.775
H   41.302   -1.631   29.221
H   35.675   -0.086   24.691
H   33.684   1.987   24.415
H   35.799   2.828   24.048
H   36.805   1.505   23.361
H   37.163   2.267   24.994
H   33.770   -0.409   23.821
H   32.182   0.305   24.374
H   33.656   -2.142   24.961
H   32.096   -1.628   25.549
H   33.680   -1.144   26.446
H   29.299   2.040   27.485
H   29.500   3.774   27.231
H   28.772   3.162   28.728
H   30.980   2.480   33.033
H   30.782   5.701   35.970
H   32.399   4.954   36.046
H   31.055   4.171   36.826
H   35.732   3.416   38.922
H   36.960   3.826   37.655
H   38.298   1.944   38.412
H   36.024   1.547   41.713
H   35.732   3.211   41.102
H   37.034   2.856   42.295
H   39.001   0.455   40.100
H   38.035   0.459   41.534
H   39.785   2.877   41.458
H   40.685   1.426   40.977
H   39.114   0.846   43.265
H   40.282   2.180   43.674
H   41.642   -0.080   42.406
H   40.533   -0.573   45.097
H   41.654   -1.539   44.119
H   39.990   -1.348   43.680
H   42.550   0.911   45.042
H   42.314   2.306   44.061
H   43.636   1.135   42.188
H   44.060   -0.008   43.403
H   45.441   1.209   44.329
H   44.259   2.366   44.753
H   44.791   2.970   41.985
H   47.418   3.666   42.942
H   47.022   2.645   41.563
H   47.204   1.918   43.177
H   45.819   4.622   44.203
H   44.067   4.456   44.093
H   44.824   5.058   41.343
H   46.216   5.594   42.224
H   44.164   6.768   43.692
H   43.412   6.609   42.123
H   45.890   7.521   41.426
H   47.196   8.428   42.964
H   45.990   8.163   44.233
H   45.999   9.684   43.380
H   44.584   10.026   41.856
H   43.273   8.808   41.588
H   44.493   9.006   40.393
