%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_201_chromophore_20 TDDFT with cam-b3lyp functional

0 1
Mg   7.066   57.131   41.377
C   6.326   53.800   41.126
C   10.232   56.482   40.316
C   7.749   60.518   41.098
C   3.773   57.830   42.098
N   8.069   55.369   40.650
C   7.569   54.082   40.608
C   8.543   53.114   39.913
C   9.890   53.851   40.073
C   9.377   55.358   40.394
C   10.841   53.240   41.064
C   8.118   52.861   38.440
C   8.396   51.453   37.782
C   9.410   51.145   36.718
O   10.206   50.252   36.731
O   9.036   51.886   35.619
N   8.783   58.321   40.849
C   10.020   57.840   40.552
C   10.967   58.965   40.362
C   10.239   60.189   40.613
C   8.901   59.721   40.859
C   12.417   58.839   40.110
C   10.804   61.606   40.564
O   12.012   61.867   40.364
C   9.906   62.897   40.717
N   5.900   58.937   41.391
C   6.409   60.176   41.384
C   5.347   61.246   41.720
C   4.071   60.393   42.076
C   4.586   58.962   41.810
C   5.736   62.282   42.830
C   2.854   60.826   41.219
C   1.401   60.771   41.856
N   5.295   56.045   41.577
C   4.026   56.462   41.931
C   3.311   55.225   42.334
C   4.191   54.165   42.021
C   5.289   54.738   41.545
C   1.999   55.105   43.117
C   4.352   52.722   41.971
O   3.734   51.742   42.404
C   5.728   52.445   41.244
C   5.409   51.764   39.988
O   4.697   52.220   39.114
O   6.024   50.519   39.874
C   5.548   49.731   38.745
C   9.691   51.459   34.367
C   9.263   52.401   33.219
C   8.045   52.675   32.636
C   6.769   51.850   32.873
C   8.009   53.706   31.501
C   7.390   55.070   31.799
C   7.994   56.183   30.860
C   9.100   57.081   31.417
C   8.604   58.521   31.802
C   10.325   57.192   30.462
C   11.755   57.250   31.130
C   12.687   58.304   30.528
C   13.650   57.820   29.451
C   15.154   57.885   29.938
C   13.365   58.496   28.049
C   14.141   57.884   26.826
C   13.366   57.565   25.534
C   13.855   58.564   24.502
C   12.990   59.874   24.437
C   13.775   57.877   23.104
H   11.266   56.288   40.023
H   7.834   61.601   40.984
H   2.773   57.998   42.501
H   8.506   52.172   40.459
H   10.543   53.777   39.203
H   10.231   52.784   41.844
H   11.431   53.987   41.593
H   11.537   52.498   40.673
H   8.579   53.653   37.851
H   7.035   52.851   38.319
H   7.429   51.177   37.361
H   8.519   50.633   38.491
H   12.581   59.064   39.056
H   12.961   57.950   40.432
H   12.796   59.634   40.753
H   9.448   62.829   41.704
H   9.161   63.095   39.947
H   10.518   63.795   40.629
H   5.299   61.704   40.732
H   3.788   60.464   43.126
H   5.808   63.318   42.499
H   6.706   61.974   43.220
H   4.985   62.149   43.608
H   2.797   60.218   40.316
H   2.895   61.861   40.878
H   0.941   61.667   41.440
H   1.402   60.776   42.946
H   0.835   59.896   41.536
H   1.960   55.704   44.027
H   1.721   54.083   43.378
H   1.209   55.513   42.487
H   6.331   51.772   41.853
H   4.642   49.199   39.036
H   6.227   48.956   38.388
H   5.224   50.144   37.791
H   9.593   50.392   34.170
H   10.750   51.537   34.614
H   10.034   53.137   32.992
H   6.428   51.259   32.023
H   5.942   52.493   33.174
H   6.948   51.209   33.736
H   7.389   53.227   30.743
H   9.028   53.823   31.132
H   7.545   55.259   32.861
H   6.316   55.006   31.629
H   7.186   56.853   30.565
H   8.287   55.741   29.907
H   9.470   56.563   32.302
H   9.210   59.045   32.541
H   7.677   58.518   32.376
H   8.478   59.163   30.930
H   10.150   57.900   29.652
H   10.341   56.233   29.945
H   12.288   56.304   31.029
H   11.644   57.447   32.196
H   13.134   58.875   31.342
H   12.083   59.070   30.043
H   13.387   56.768   29.342
H   15.470   56.908   30.305
H   15.265   58.575   30.776
H   15.805   58.132   29.100
H   13.637   59.527   28.274
H   12.309   58.350   27.819
H   14.678   57.004   27.181
H   14.974   58.541   26.578
H   12.295   57.613   25.731
H   13.666   56.539   25.325
H   14.816   59.063   24.631
H   13.624   60.700   24.114
H   12.494   60.014   25.397
H   12.260   59.788   23.632
H   12.774   57.494   22.906
H   14.443   57.023   23.218
H   14.105   58.662   22.424

