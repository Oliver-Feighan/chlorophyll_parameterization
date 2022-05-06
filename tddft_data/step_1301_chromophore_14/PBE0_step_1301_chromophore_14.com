%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1301_chromophore_14 TDDFT with PBE1PBE functional

0 1
Mg   45.627   44.497   43.170
C   42.196   44.148   42.958
C   46.007   41.232   42.482
C   48.846   45.120   42.817
C   45.054   48.006   43.425
N   44.223   42.927   42.833
C   42.904   42.935   42.958
C   42.336   41.488   42.790
C   43.595   40.592   42.896
C   44.691   41.666   42.832
C   43.650   39.812   44.237
C   41.478   41.339   41.587
C   41.867   42.078   40.285
C   41.753   41.362   38.973
O   41.540   40.147   38.816
O   41.929   42.276   37.949
N   47.243   43.300   42.690
C   47.180   41.968   42.520
C   48.523   41.461   42.322
C   49.434   42.551   42.390
C   48.519   43.733   42.626
C   48.896   40.044   42.045
C   50.876   42.511   42.180
O   51.522   41.459   42.114
C   51.693   43.752   42.225
N   46.827   46.331   43.193
C   48.134   46.316   43.023
C   48.688   47.787   42.951
C   47.437   48.587   43.368
C   46.370   47.580   43.373
C   49.977   47.990   43.840
C   47.283   49.692   42.300
C   46.739   50.977   42.744
N   43.999   45.896   43.234
C   43.904   47.245   43.335
C   42.522   47.686   43.354
C   41.803   46.469   43.275
C   42.707   45.424   43.202
C   42.037   49.104   43.561
C   40.394   45.925   43.181
O   39.315   46.472   43.274
C   40.654   44.353   42.901
C   39.821   43.566   43.828
O   38.928   42.801   43.538
O   40.263   43.839   45.116
C   39.903   42.742   45.954
C   41.642   41.942   36.551
C   42.816   42.169   35.712
C   43.622   41.347   34.954
C   43.665   39.781   35.106
C   44.747   41.972   34.097
C   44.676   41.962   32.569
C   43.910   43.193   32.129
C   44.281   43.693   30.660
C   44.519   45.228   30.598
C   43.231   43.266   29.620
C   43.892   42.486   28.510
C   42.999   42.670   27.274
C   43.785   43.155   26.008
C   42.950   44.247   25.230
C   44.208   41.981   25.117
C   45.664   42.034   24.496
C   45.710   41.892   22.981
C   47.165   41.752   22.541
C   47.237   42.250   21.093
C   47.755   40.365   22.710
H   46.026   40.162   42.263
H   49.873   45.468   42.693
H   44.807   49.041   43.670
H   41.690   41.254   43.636
H   43.714   39.889   42.072
H   42.896   39.917   45.017
H   44.401   40.299   44.858
H   43.905   38.779   44.001
H   40.482   41.703   41.840
H   41.449   40.256   41.471
H   42.899   42.400   40.422
H   41.265   42.969   40.104
H   48.034   39.431   42.306
H   49.707   39.625   42.641
H   49.094   39.981   40.975
H   51.691   44.151   43.239
H   51.227   44.460   41.539
H   52.723   43.466   42.013
H   48.990   47.814   41.904
H   47.639   49.008   44.353
H   49.859   48.855   44.491
H   50.804   48.190   43.158
H   50.213   47.137   44.477
H   46.658   49.295   41.500
H   48.270   49.930   41.903
H   47.562   51.513   43.218
H   45.889   50.871   43.417
H   46.433   51.633   41.928
H   42.759   49.821   43.950
H   41.221   49.009   44.278
H   41.779   49.480   42.571
H   40.296   44.215   41.881
H   40.396   42.701   46.926
H   40.247   41.815   45.495
H   38.831   42.791   46.144
H   40.762   42.500   36.231
H   41.251   40.934   36.412
H   43.015   43.241   35.716
H   43.609   39.353   34.105
H   42.915   39.321   35.749
H   44.675   39.424   35.308
H   45.751   41.592   34.288
H   44.917   42.982   34.471
H   44.151   41.106   32.144
H   45.607   41.837   32.017
H   44.023   43.971   32.884
H   42.832   43.032   32.106
H   45.302   43.339   30.515
H   44.597   45.659   29.600
H   45.446   45.575   31.053
H   43.713   45.609   31.226
H   42.771   44.191   29.272
H   42.475   42.685   30.149
H   43.950   41.411   28.682
H   44.918   42.727   28.234
H   42.065   43.199   27.462
H   42.655   41.636   27.235
H   44.677   43.622   26.426
H   41.895   44.312   25.497
H   43.000   44.170   24.143
H   43.421   45.206   25.444
H   43.540   41.936   24.257
H   44.086   41.073   25.708
H   46.305   41.318   25.009
H   46.068   43.006   24.780
H   45.282   42.696   22.383
H   45.185   41.020   22.591
H   47.723   42.488   23.120
H   46.755   41.518   20.445
H   48.244   42.398   20.704
H   46.812   43.249   20.997
H   48.717   40.432   23.218
H   47.997   39.926   21.742
H   47.118   39.728   23.323

