%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1_chromophore_9 TDDFT with blyp functional

0 1
Mg   35.716   1.255   29.884
C   33.260   2.269   32.062
C   37.975   1.184   32.632
C   38.346   0.782   27.813
C   33.670   1.646   27.255
N   35.616   1.687   32.102
C   34.475   2.103   32.734
C   34.689   2.203   34.246
C   36.276   2.035   34.333
C   36.659   1.612   32.945
C   37.029   3.302   34.882
C   33.912   1.084   35.079
C   33.167   1.390   36.356
C   33.843   2.221   37.388
O   33.421   3.341   37.728
O   34.933   1.640   37.876
N   37.887   0.958   30.157
C   38.550   0.788   31.322
C   39.891   0.361   31.101
C   40.027   0.200   29.686
C   38.730   0.727   29.166
C   40.802   0.224   32.278
C   41.264   -0.169   28.885
O   41.334   -0.005   27.663
C   42.484   -0.744   29.583
N   36.020   1.367   27.779
C   37.178   1.001   27.169
C   36.981   0.891   25.658
C   35.415   1.001   25.519
C   34.996   1.352   26.933
C   37.785   1.788   24.671
C   34.665   -0.246   24.889
C   34.104   -1.334   25.780
N   33.899   1.922   29.647
C   33.078   1.916   28.548
C   31.778   2.415   28.813
C   31.777   2.525   30.200
C   33.078   2.193   30.658
C   30.562   2.677   27.934
C   31.029   2.875   31.327
O   29.887   3.294   31.372
C   31.955   2.758   32.664
C   32.020   4.078   33.279
O   32.548   5.092   32.781
O   31.387   4.147   34.480
C   31.100   5.466   35.055
C   35.582   2.283   39.059
C   36.956   1.752   39.281
C   37.663   1.567   40.454
C   37.082   1.906   41.858
C   39.057   0.996   40.450
C   39.271   -0.353   41.110
C   40.193   -0.233   42.373
C   41.643   -0.440   42.059
C   42.006   -1.929   42.095
C   42.605   0.519   42.836
C   43.418   1.449   41.910
C   44.351   2.278   42.795
C   44.666   3.666   42.143
C   46.050   3.677   41.413
C   44.807   4.773   43.233
C   43.974   6.069   42.948
C   44.742   7.371   43.228
C   44.525   8.505   42.286
C   45.786   8.771   41.424
C   43.991   9.797   43.027
H   38.549   1.115   33.558
H   39.099   0.356   27.147
H   32.904   1.431   26.508
H   34.281   3.155   34.588
H   36.555   1.259   35.046
H   37.568   3.002   35.780
H   36.247   4.051   35.008
H   37.838   3.673   34.252
H   34.722   0.389   35.301
H   33.307   0.459   34.423
H   32.770   0.470   36.785
H   32.335   2.027   36.058
H   41.314   -0.725   32.123
H   40.230   0.117   33.200
H   41.529   1.024   32.422
H   42.250   -1.487   30.346
H   43.158   0.065   29.868
H   42.999   -1.300   28.799
H   37.387   -0.095   25.431
H   35.108   1.750   24.790
H   37.128   2.526   24.211
H   38.164   1.122   23.896
H   38.578   2.241   25.265
H   35.371   -0.715   24.205
H   33.813   0.096   24.301
H   34.443   -2.324   25.474
H   33.014   -1.347   25.820
H   34.482   -1.099   26.775
H   30.791   2.376   26.911
H   30.304   3.735   27.987
H   29.727   2.091   28.317
H   31.326   2.109   33.274
H   30.272   5.471   35.764
H   31.016   6.283   34.339
H   31.936   5.664   35.726
H   35.108   1.954   39.984
H   35.775   3.342   38.890
H   37.467   1.447   38.368
H   37.873   2.264   42.516
H   36.720   1.008   42.359
H   36.270   2.623   41.733
H   39.772   1.671   40.919
H   39.282   0.849   39.393
H   39.536   -1.174   40.445
H   38.274   -0.648   41.439
H   39.775   -1.003   43.021
H   40.001   0.673   42.948
H   41.848   -0.157   41.027
H   42.301   -2.254   41.097
H   41.147   -2.584   42.237
H   42.722   -2.184   42.877
H   43.228   -0.037   43.537
H   41.945   1.215   43.355
H   42.623   2.041   41.455
H   44.030   0.926   41.175
H   45.262   1.704   42.962
H   44.003   2.425   43.818
H   43.904   3.914   41.404
H   46.194   4.660   40.965
H   45.936   2.868   40.692
H   46.743   3.305   42.168
H   45.840   5.106   43.335
H   44.487   4.384   44.200
H   43.057   6.043   43.535
H   43.678   6.089   41.899
H   45.801   7.203   43.426
H   44.400   7.671   44.219
H   43.757   8.162   41.593
H   45.362   9.038   40.457
H   46.351   7.847   41.306
H   46.401   9.604   41.767
H   44.493   10.048   43.961
H   42.917   9.926   43.167
H   44.126   10.671   42.389

