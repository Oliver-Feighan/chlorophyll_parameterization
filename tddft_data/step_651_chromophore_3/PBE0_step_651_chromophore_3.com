%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_651_chromophore_3 TDDFT with PBE1PBE functional

0 1
Mg   2.056   8.008   26.355
C   2.087   9.788   29.327
C   2.508   5.144   28.119
C   2.393   6.607   23.487
C   1.910   11.197   24.637
N   2.336   7.571   28.500
C   2.050   8.418   29.570
C   1.943   7.692   30.846
C   2.446   6.253   30.437
C   2.473   6.306   28.909
C   3.804   5.784   31.084
C   0.597   7.837   31.568
C   0.518   8.217   33.105
C   1.800   7.804   33.887
O   2.887   8.324   33.941
O   1.440   6.757   34.688
N   2.393   6.112   25.817
C   2.443   5.050   26.704
C   2.547   3.876   25.958
C   2.531   4.253   24.565
C   2.461   5.680   24.540
C   2.558   2.541   26.662
C   2.669   3.363   23.390
O   2.754   3.854   22.264
C   2.691   1.812   23.552
N   1.894   8.799   24.401
C   2.193   7.993   23.320
C   2.057   8.726   21.959
C   1.444   10.054   22.448
C   1.884   10.040   23.900
C   3.480   8.861   21.335
C   -0.100   10.077   22.390
C   -0.741   10.789   21.191
N   2.201   10.113   26.834
C   2.144   11.233   26.016
C   2.144   12.385   26.925
C   2.175   11.823   28.199
C   2.231   10.482   28.089
C   2.204   13.876   26.504
C   2.240   12.196   29.591
O   2.247   13.284   30.111
C   2.137   10.828   30.393
C   3.333   10.782   31.292
O   4.489   10.392   30.992
O   2.987   11.326   32.535
C   4.040   11.565   33.453
C   2.270   6.323   35.782
C   1.547   5.223   36.443
C   1.794   4.737   37.648
C   3.063   4.996   38.533
C   0.699   3.926   38.402
C   0.611   2.439   37.979
C   0.553   1.468   39.157
C   1.724   0.414   39.078
C   3.079   1.109   39.429
C   1.278   -0.770   40.136
C   1.809   -2.175   39.739
C   2.286   -3.064   40.946
C   3.716   -3.735   40.697
C   3.624   -5.304   40.843
C   4.785   -3.090   41.643
C   5.994   -2.530   40.823
C   6.913   -1.475   41.467
C   8.359   -1.394   40.856
C   9.423   -1.201   41.991
C   8.514   -0.160   39.938
H   2.764   4.253   28.695
H   2.544   6.027   22.574
H   1.877   12.144   24.093
H   2.763   8.053   31.468
H   1.679   5.536   30.727
H   3.669   4.738   31.359
H   4.161   6.450   31.869
H   4.594   5.754   30.334
H   -0.021   6.941   31.512
H   0.061   8.641   31.065
H   -0.369   7.711   33.484
H   0.343   9.284   33.246
H   1.634   2.007   26.444
H   2.617   2.484   27.749
H   3.458   2.053   26.288
H   1.628   1.662   23.740
H   3.277   1.461   24.401
H   2.945   1.475   22.547
H   1.414   8.152   21.292
H   1.885   10.810   21.798
H   3.711   7.887   20.904
H   4.146   9.038   22.180
H   3.548   9.655   20.592
H   -0.490   10.672   23.216
H   -0.495   9.093   22.645
H   -0.782   9.999   20.441
H   -0.134   11.690   21.098
H   -1.764   10.977   21.517
H   1.677   13.828   25.551
H   3.248   14.171   26.400
H   1.689   14.488   27.243
H   1.190   10.776   30.932
H   5.042   11.245   33.169
H   3.685   11.211   34.421
H   3.987   12.641   33.619
H   2.579   7.051   36.532
H   3.241   6.066   35.357
H   0.489   5.004   36.297
H   2.705   5.795   39.183
H   3.829   5.468   37.919
H   3.494   4.071   38.916
H   -0.250   4.428   38.215
H   0.843   4.011   39.479
H   1.390   2.226   37.248
H   -0.351   2.507   37.471
H   -0.361   0.934   38.898
H   0.537   1.895   40.160
H   1.766   0.034   38.057
H   2.769   1.960   40.036
H   3.471   1.550   38.512
H   3.780   0.520   40.020
H   0.203   -0.763   40.317
H   1.648   -0.489   41.122
H   2.620   -2.114   39.013
H   0.897   -2.688   39.435
H   1.602   -3.901   41.088
H   2.399   -2.462   41.847
H   4.003   -3.624   39.651
H   2.846   -5.595   41.549
H   4.596   -5.690   41.150
H   3.509   -5.672   39.824
H   5.168   -3.906   42.257
H   4.418   -2.215   42.180
H   5.590   -2.109   39.902
H   6.595   -3.351   40.435
H   6.916   -1.630   42.546
H   6.385   -0.532   41.327
H   8.598   -2.297   40.294
H   9.979   -2.107   42.232
H   8.953   -0.772   42.877
H   10.155   -0.415   41.802
H   9.245   -0.413   39.171
H   8.983   0.699   40.420
H   7.601   0.227   39.486

