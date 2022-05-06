%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1651_chromophore_1 TDDFT with blyp functional

0 1
Mg   -1.892   17.407   26.252
C   -2.429   15.034   28.954
C   -2.207   19.902   28.466
C   -1.907   19.444   23.637
C   -1.899   14.679   24.121
N   -2.244   17.448   28.437
C   -2.459   16.428   29.383
C   -2.646   16.993   30.843
C   -2.746   18.480   30.554
C   -2.356   18.677   29.082
C   -4.098   19.148   30.904
C   -1.444   16.699   31.774
C   -1.524   17.270   33.230
C   -1.174   16.444   34.464
O   -1.785   15.454   34.963
O   -0.027   17.064   35.061
N   -1.850   19.463   26.036
C   -1.938   20.328   27.080
C   -1.972   21.676   26.572
C   -1.735   21.547   25.163
C   -1.859   20.152   24.846
C   -2.124   22.879   27.486
C   -1.464   22.683   24.146
O   -1.311   22.468   22.964
C   -1.274   24.089   24.655
N   -1.843   17.141   24.233
C   -2.020   18.105   23.323
C   -2.236   17.679   21.883
C   -1.992   16.134   22.012
C   -1.865   15.972   23.580
C   -3.561   18.030   21.136
C   -0.728   15.639   21.289
C   0.404   15.316   22.080
N   -2.267   15.324   26.438
C   -2.089   14.364   25.451
C   -2.239   13.012   25.994
C   -2.375   13.220   27.384
C   -2.313   14.644   27.638
C   -2.343   11.844   25.105
C   -2.396   12.583   28.632
O   -2.177   11.424   29.039
C   -2.432   13.698   29.720
C   -1.298   13.438   30.658
O   -0.154   13.799   30.489
O   -1.765   12.849   31.843
C   -0.760   12.463   32.839
C   0.304   16.585   36.450
C   -0.314   17.495   37.611
C   -0.608   17.208   38.968
C   -0.221   15.926   39.575
C   -1.223   18.302   39.841
C   -0.302   19.039   40.948
C   1.038   19.482   40.366
C   1.525   20.877   40.898
C   1.326   21.988   39.771
C   3.003   20.954   41.420
C   3.243   22.286   42.151
C   4.386   22.363   43.257
C   3.853   22.847   44.638
C   4.953   23.688   45.397
C   3.230   21.702   45.510
C   1.896   22.103   46.138
C   0.583   21.547   45.422
C   -0.768   22.035   46.144
C   -1.348   21.153   47.184
C   -1.759   22.319   44.968
H   -2.333   20.755   29.135
H   -2.027   20.144   22.807
H   -1.905   13.789   23.489
H   -3.507   16.603   31.387
H   -2.103   19.099   31.180
H   -3.984   19.844   31.735
H   -4.824   18.341   31.001
H   -4.383   19.688   30.001
H   -0.526   16.956   31.246
H   -1.377   15.639   32.017
H   -2.553   17.600   33.372
H   -1.030   18.241   33.209
H   -2.633   23.649   26.906
H   -1.222   23.286   27.943
H   -2.732   22.550   28.329
H   -2.271   24.484   24.851
H   -0.690   24.613   23.899
H   -0.737   24.035   25.601
H   -1.390   18.146   21.380
H   -2.836   15.544   21.653
H   -4.005   17.197   20.590
H   -3.421   18.753   20.332
H   -4.325   18.425   21.805
H   -0.321   16.380   20.601
H   -0.931   14.794   20.630
H   1.314   15.702   21.620
H   0.409   14.226   22.081
H   0.473   15.867   23.018
H   -1.440   11.753   24.502
H   -3.109   12.130   24.384
H   -2.450   10.989   25.773
H   -3.365   13.638   30.280
H   -0.545   13.435   33.282
H   0.115   12.014   32.370
H   -1.284   11.792   33.520
H   1.385   16.704   36.529
H   0.114   15.537   36.680
H   -0.973   18.248   37.178
H   -1.060   15.572   40.174
H   0.627   16.017   40.255
H   -0.001   15.127   38.867
H   -1.998   17.751   40.373
H   -1.639   19.066   39.185
H   -0.082   18.250   41.667
H   -0.820   19.801   41.530
H   0.895   19.410   39.288
H   1.830   18.767   40.585
H   0.870   21.345   41.632
H   0.791   21.648   38.884
H   2.277   22.384   39.417
H   0.743   22.846   40.108
H   3.664   20.814   40.564
H   3.189   20.231   42.214
H   2.323   22.606   42.641
H   3.529   23.063   41.442
H   5.209   22.972   42.884
H   4.703   21.328   43.385
H   3.020   23.466   44.305
H   5.902   23.566   44.874
H   5.051   23.372   46.435
H   4.778   24.760   45.486
H   3.892   21.615   46.372
H   3.274   20.680   45.131
H   1.915   23.191   46.200
H   1.757   21.685   47.135
H   0.681   20.462   45.456
H   0.613   21.812   44.365
H   -0.476   22.927   46.699
H   -0.688   20.296   47.320
H   -2.314   20.728   46.909
H   -1.476   21.653   48.144
H   -1.461   22.101   43.942
H   -1.860   23.402   45.045
H   -2.686   21.848   45.296

