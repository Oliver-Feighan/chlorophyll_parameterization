%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1851_chromophore_23 TDDFT with PBE1PBE functional

0 1
Mg   -9.157   41.095   42.309
C   -9.008   37.711   41.429
C   -6.063   41.562   40.844
C   -9.698   44.295   42.449
C   -12.506   40.532   43.451
N   -7.822   39.776   41.128
C   -7.960   38.461   40.802
C   -6.707   37.961   40.004
C   -5.650   39.057   40.204
C   -6.548   40.249   40.809
C   -4.549   38.919   41.293
C   -7.054   37.613   38.485
C   -6.540   36.276   38.041
C   -5.454   36.306   36.857
O   -4.230   36.332   37.064
O   -6.053   36.572   35.576
N   -7.951   42.752   41.788
C   -6.778   42.720   41.228
C   -6.220   44.066   41.135
C   -7.223   44.953   41.629
C   -8.392   44.037   41.930
C   -4.699   44.376   40.849
C   -7.110   46.475   41.617
O   -6.173   47.085   41.043
C   -8.118   47.346   42.309
N   -10.985   42.238   42.675
C   -10.836   43.554   42.785
C   -12.193   44.179   43.305
C   -13.106   42.954   43.679
C   -12.113   41.849   43.258
C   -11.936   45.226   44.500
C   -14.578   42.952   43.089
C   -15.616   43.030   44.187
N   -10.482   39.479   42.599
C   -11.735   39.367   43.076
C   -12.118   37.947   43.169
C   -11.053   37.300   42.520
C   -10.121   38.258   42.156
C   -13.323   37.341   43.909
C   -10.528   35.937   42.188
O   -10.873   34.791   42.516
C   -9.302   36.188   41.252
C   -8.123   35.180   41.599
O   -7.736   34.813   42.708
O   -7.753   34.626   40.408
C   -6.835   33.524   40.552
C   -5.204   36.937   34.426
C   -5.875   36.658   33.079
C   -6.739   37.356   32.267
C   -7.229   38.767   32.604
C   -7.241   36.714   30.998
C   -6.299   36.333   29.826
C   -7.062   36.114   28.467
C   -6.593   36.987   27.301
C   -5.936   36.120   26.196
C   -7.665   37.889   26.591
C   -7.951   39.315   27.311
C   -9.458   39.385   27.706
C   -9.684   39.684   29.265
C   -10.855   38.914   29.894
C   -9.772   41.269   29.535
C   -8.570   41.892   30.389
C   -8.470   43.447   30.261
C   -9.073   44.078   31.527
C   -10.364   44.848   31.325
C   -8.070   44.961   32.369
H   -5.087   41.798   40.414
H   -9.856   45.365   42.602
H   -13.467   40.264   43.895
H   -6.310   37.185   40.659
H   -5.184   39.351   39.264
H   -4.708   38.043   41.922
H   -4.514   39.750   41.998
H   -3.555   38.858   40.851
H   -6.774   38.375   37.757
H   -8.096   37.452   38.209
H   -7.290   35.591   37.648
H   -6.057   35.726   38.849
H   -4.462   43.934   39.881
H   -4.069   43.950   41.629
H   -4.452   45.435   40.778
H   -8.209   46.948   43.319
H   -9.056   47.306   41.755
H   -7.791   48.373   42.474
H   -12.478   44.774   42.437
H   -13.199   42.897   44.763
H   -10.938   44.984   44.865
H   -12.728   45.112   45.241
H   -11.965   46.231   44.080
H   -14.735   42.012   42.561
H   -14.716   43.804   42.422
H   -15.099   43.297   45.108
H   -16.088   42.063   44.366
H   -16.370   43.762   43.897
H   -12.866   36.633   44.601
H   -14.118   36.844   43.353
H   -13.722   38.155   44.515
H   -9.596   36.086   40.207
H   -7.092   32.924   39.679
H   -6.834   32.932   41.467
H   -5.862   34.002   40.435
H   -4.233   36.442   34.411
H   -4.921   37.985   34.527
H   -5.550   35.680   32.725
H   -7.051   39.320   31.682
H   -6.569   39.180   33.367
H   -8.276   38.911   32.871
H   -8.099   37.280   30.635
H   -7.679   35.791   31.377
H   -5.884   35.361   30.091
H   -5.548   37.123   29.792
H   -8.137   36.257   28.577
H   -6.884   35.079   28.176
H   -5.775   37.666   27.541
H   -6.112   36.710   25.296
H   -6.461   35.165   26.213
H   -4.863   36.026   26.364
H   -8.580   37.304   26.494
H   -7.373   38.189   25.584
H   -7.739   40.032   26.518
H   -7.265   39.402   28.153
H   -9.999   38.490   27.400
H   -9.934   40.195   27.153
H   -8.801   39.335   29.800
H   -10.450   37.998   30.324
H   -11.763   38.877   29.294
H   -11.239   39.433   30.773
H   -10.684   41.505   30.083
H   -9.994   41.736   28.575
H   -7.670   41.476   29.938
H   -8.661   41.499   31.402
H   -9.011   43.815   29.390
H   -7.442   43.723   30.024
H   -9.332   43.250   32.187
H   -11.135   44.296   30.787
H   -10.159   45.767   30.776
H   -10.756   45.107   32.309
H   -7.890   44.426   33.302
H   -8.534   45.908   32.644
H   -7.160   45.097   31.783

