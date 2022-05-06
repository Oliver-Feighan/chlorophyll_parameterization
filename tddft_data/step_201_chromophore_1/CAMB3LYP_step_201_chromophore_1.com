%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_201_chromophore_1 TDDFT with cam-b3lyp functional

0 1
Mg   -1.232   17.266   26.730
C   -1.644   15.091   29.386
C   -1.738   19.931   28.936
C   -1.171   19.364   24.157
C   -1.430   14.654   24.539
N   -1.657   17.440   28.893
C   -1.755   16.415   29.832
C   -1.902   16.963   31.294
C   -2.189   18.482   31.019
C   -1.880   18.679   29.531
C   -3.602   19.020   31.465
C   -0.709   16.687   32.286
C   -0.683   17.600   33.549
C   -0.584   16.952   34.940
O   -1.216   15.907   35.251
O   0.141   17.601   35.868
N   -1.295   19.313   26.584
C   -1.373   20.258   27.624
C   -1.231   21.527   27.076
C   -1.071   21.430   25.651
C   -1.080   19.976   25.416
C   -1.362   22.802   27.942
C   -1.010   22.585   24.668
O   -0.975   22.417   23.467
C   -0.782   24.022   25.076
N   -1.257   17.113   24.677
C   -1.305   18.109   23.797
C   -1.525   17.641   22.355
C   -1.241   16.071   22.483
C   -1.291   15.917   24.003
C   -2.875   17.967   21.836
C   0.156   15.619   22.014
C   0.419   15.813   20.523
N   -1.341   15.287   26.819
C   -1.381   14.365   25.944
C   -1.367   13.041   26.518
C   -1.472   13.249   27.848
C   -1.550   14.679   28.015
C   -1.359   11.840   25.654
C   -1.636   12.610   29.122
O   -1.748   11.406   29.378
C   -1.825   13.799   30.171
C   -0.932   13.433   31.245
O   0.257   13.645   31.341
O   -1.697   12.853   32.289
C   -0.806   12.415   33.432
C   0.126   17.049   37.209
C   0.005   18.245   37.998
C   0.069   18.324   39.295
C   0.185   17.107   40.205
C   0.220   19.681   39.950
C   1.592   19.994   40.578
C   1.773   21.496   41.020
C   2.962   22.286   40.491
C   2.822   22.419   38.961
C   4.397   21.724   40.897
C   4.848   22.240   42.271
C   5.111   21.091   43.331
C   4.789   21.490   44.742
C   6.055   21.548   45.602
C   3.628   20.673   45.307
C   2.840   21.373   46.422
C   1.319   21.260   46.194
C   0.785   22.401   45.305
C   0.721   23.685   46.153
C   -0.478   22.124   44.439
H   -1.989   20.777   29.580
H   -1.137   20.085   23.337
H   -1.416   13.784   23.879
H   -2.838   16.706   31.790
H   -1.556   19.198   31.545
H   -3.946   19.679   30.668
H   -3.438   19.598   32.374
H   -4.385   18.331   31.780
H   0.183   16.855   31.682
H   -0.731   15.687   32.717
H   -1.582   18.205   33.666
H   0.184   18.258   33.489
H   -2.281   23.284   27.607
H   -0.473   23.410   27.776
H   -1.356   22.629   29.019
H   0.035   24.151   25.785
H   -1.747   24.336   25.473
H   -0.403   24.475   24.160
H   -0.697   18.117   21.831
H   -1.921   15.370   21.999
H   -3.102   18.956   22.234
H   -3.565   17.147   22.036
H   -2.812   18.112   20.757
H   0.317   14.572   22.273
H   0.801   16.311   22.555
H   0.625   14.879   19.999
H   1.280   16.449   20.318
H   -0.419   16.269   19.995
H   -2.309   11.786   25.121
H   -1.289   11.038   26.388
H   -0.588   11.652   24.906
H   -2.861   13.605   30.450
H   -0.427   13.217   34.066
H   0.043   11.773   33.200
H   -1.476   11.947   34.152
H   1.026   16.468   37.412
H   -0.766   16.459   37.418
H   -0.103   19.214   37.511
H   1.155   16.998   40.690
H   -0.099   16.159   39.749
H   -0.512   17.235   41.033
H   -0.349   19.676   40.879
H   -0.176   20.557   39.437
H   2.423   19.729   39.925
H   1.576   19.362   41.465
H   1.787   21.394   42.105
H   0.895   22.070   40.721
H   2.950   23.262   40.978
H   3.782   22.210   38.490
H   2.549   23.438   38.686
H   2.001   21.865   38.506
H   5.207   21.979   40.213
H   4.393   20.635   40.940
H   4.160   22.981   42.679
H   5.875   22.600   42.215
H   6.154   20.780   43.261
H   4.612   20.177   43.009
H   4.382   22.496   44.837
H   6.006   22.360   46.327
H   6.947   21.735   45.004
H   6.212   20.678   46.239
H   4.094   19.908   45.928
H   3.061   20.194   44.509
H   3.177   22.399   46.570
H   3.100   20.863   47.350
H   0.788   21.049   47.123
H   1.093   20.414   45.545
H   1.520   22.685   44.552
H   1.651   23.859   46.694
H   -0.063   23.502   46.888
H   0.619   24.577   45.535
H   -0.286   22.312   43.382
H   -1.203   22.910   44.648
H   -0.837   21.124   44.682

