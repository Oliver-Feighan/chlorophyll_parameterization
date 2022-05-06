%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_801_chromophore_25 TDDFT with PBE1PBE functional

0 1
Mg   -2.767   34.947   26.475
C   -3.603   33.400   29.505
C   -0.885   37.204   28.269
C   -2.455   36.833   23.678
C   -5.207   32.949   24.813
N   -2.364   35.193   28.685
C   -2.872   34.528   29.747
C   -2.310   35.182   31.096
C   -1.372   36.324   30.661
C   -1.581   36.302   29.119
C   -1.559   37.663   31.300
C   -1.617   34.170   32.102
C   -2.262   34.220   33.592
C   -1.364   34.168   34.807
O   -0.327   33.515   34.898
O   -1.683   34.963   35.882
N   -1.647   36.608   25.987
C   -0.983   37.393   26.896
C   -0.397   38.523   26.245
C   -0.802   38.487   24.802
C   -1.690   37.272   24.781
C   0.390   39.494   27.021
C   -0.495   39.348   23.572
O   -0.831   39.093   22.423
C   0.303   40.625   23.774
N   -3.649   34.787   24.574
C   -3.295   35.737   23.536
C   -4.183   35.472   22.236
C   -4.696   34.003   22.541
C   -4.573   33.958   24.075
C   -5.254   36.594   21.998
C   -3.891   32.838   21.848
C   -4.286   32.637   20.394
N   -4.128   33.478   26.981
C   -4.988   32.701   26.190
C   -5.530   31.627   26.989
C   -5.052   31.887   28.281
C   -4.164   32.997   28.227
C   -6.514   30.566   26.558
C   -5.052   31.402   29.598
O   -5.519   30.405   30.099
C   -4.353   32.477   30.513
C   -5.378   33.117   31.475
O   -6.070   34.068   31.218
O   -5.341   32.429   32.653
C   -6.102   32.986   33.788
C   -0.833   34.765   37.046
C   -1.249   35.779   38.102
C   -0.602   36.009   39.316
C   0.418   35.056   39.960
C   -1.039   37.215   40.189
C   -2.277   36.990   41.020
C   -2.241   37.823   42.377
C   -2.640   36.950   43.695
C   -4.143   37.230   44.121
C   -1.685   37.343   44.846
C   -0.761   36.176   45.426
C   0.546   36.664   46.040
C   1.323   35.468   46.726
C   0.874   35.332   48.215
C   2.809   35.644   46.674
C   3.423   34.709   45.542
C   4.019   35.648   44.492
C   4.327   35.047   43.180
C   3.116   34.767   42.275
C   5.254   33.871   43.289
H   -0.233   37.802   28.909
H   -2.398   37.271   22.679
H   -5.918   32.283   24.321
H   -3.166   35.544   31.666
H   -0.342   36.139   30.966
H   -1.635   38.489   30.593
H   -0.813   37.971   32.033
H   -2.420   37.689   31.968
H   -0.544   34.312   32.231
H   -1.735   33.152   31.731
H   -3.015   33.432   33.599
H   -2.734   35.201   33.645
H   1.007   40.098   26.354
H   1.107   39.156   27.770
H   -0.311   40.256   27.359
H   0.039   41.280   24.605
H   0.531   41.153   22.848
H   1.235   40.162   24.099
H   -3.582   35.462   21.326
H   -5.772   34.042   22.369
H   -5.165   37.268   22.850
H   -6.266   36.189   22.013
H   -5.074   37.126   21.064
H   -3.893   31.909   22.417
H   -2.846   33.114   21.708
H   -5.132   33.273   20.132
H   -4.723   31.640   20.329
H   -3.439   32.811   19.731
H   -6.103   29.556   26.534
H   -7.068   30.786   25.646
H   -7.289   30.636   27.322
H   -3.557   32.019   31.100
H   -7.013   33.431   33.388
H   -5.446   33.799   34.100
H   -6.371   32.248   34.544
H   0.222   34.942   36.838
H   -0.903   33.766   37.478
H   -2.062   36.403   37.733
H   1.421   35.358   40.263
H   0.710   34.335   39.196
H   0.026   34.519   40.824
H   -1.117   38.193   39.715
H   -0.261   37.284   40.950
H   -2.497   35.925   41.093
H   -3.110   37.327   40.402
H   -2.859   38.708   42.229
H   -1.244   38.250   42.484
H   -2.587   35.872   43.539
H   -4.271   38.006   44.876
H   -4.600   36.331   44.535
H   -4.785   37.531   43.293
H   -2.316   37.750   45.636
H   -1.009   38.157   44.584
H   -0.453   35.438   44.685
H   -1.345   35.598   46.141
H   0.333   37.449   46.765
H   1.181   37.191   45.329
H   1.085   34.477   46.338
H   0.672   34.262   48.278
H   -0.068   35.808   48.487
H   1.541   35.631   49.024
H   3.245   35.383   47.638
H   3.007   36.681   46.402
H   2.721   33.967   45.163
H   4.229   34.158   46.027
H   4.882   35.998   45.059
H   3.297   36.458   44.397
H   4.718   35.924   42.666
H   2.853   33.713   42.371
H   3.417   34.953   41.244
H   2.275   35.429   42.482
H   6.064   34.083   42.591
H   4.730   32.941   43.069
H   5.739   33.935   44.263

