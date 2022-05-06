%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1401_chromophore_8 TDDFT with cam-b3lyp functional

0 1
Mg   44.610   2.697   47.228
C   42.295   5.255   46.823
C   41.938   0.513   47.008
C   46.811   0.079   47.239
C   47.103   4.960   46.344
N   42.341   2.860   46.858
C   41.668   4.024   46.765
C   40.255   3.894   46.519
C   40.054   2.369   46.857
C   41.510   1.837   46.931
C   39.155   2.059   48.133
C   39.858   4.312   45.065
C   38.567   5.206   44.880
C   37.969   5.222   43.490
O   36.834   4.827   43.308
O   38.776   5.657   42.562
N   44.426   0.587   47.202
C   43.223   -0.071   47.093
C   43.508   -1.526   47.243
C   44.905   -1.734   47.231
C   45.470   -0.373   47.319
C   42.425   -2.488   47.111
C   45.636   -3.023   47.220
O   45.066   -4.131   47.228
C   47.030   -2.971   47.523
N   46.553   2.509   46.838
C   47.273   1.340   46.945
C   48.847   1.632   46.889
C   48.842   3.104   46.289
C   47.392   3.574   46.442
C   49.500   1.373   48.270
C   49.477   3.322   44.930
C   50.843   3.973   44.863
N   44.794   4.719   46.906
C   45.861   5.459   46.538
C   45.436   6.770   46.404
C   44.025   6.824   46.554
C   43.719   5.494   46.870
C   46.253   7.965   45.939
C   42.802   7.633   46.529
O   42.595   8.811   46.223
C   41.649   6.650   46.873
C   41.064   6.863   48.151
O   41.512   6.501   49.205
O   39.776   7.258   48.003
C   38.997   7.557   49.222
C   38.302   5.612   41.161
C   39.501   5.140   40.263
C   39.685   5.489   38.982
C   38.709   6.315   38.218
C   40.975   5.019   38.366
C   41.087   3.543   37.849
C   42.304   2.736   38.376
C   43.202   2.262   37.190
C   44.016   0.999   37.704
C   44.114   3.416   36.722
C   45.041   3.100   35.559
C   45.521   4.318   34.778
C   45.220   4.209   33.225
C   46.521   4.168   32.455
C   44.346   5.371   32.601
C   43.162   4.870   31.773
C   43.277   5.244   30.286
C   43.369   3.994   29.368
C   41.934   3.473   29.120
C   44.217   4.106   28.125
H   41.055   -0.128   46.995
H   47.668   -0.591   47.337
H   47.854   5.699   46.057
H   39.768   4.531   47.256
H   39.520   1.904   46.028
H   38.574   2.894   48.525
H   39.824   1.792   48.951
H   38.607   1.150   47.887
H   39.792   3.395   44.479
H   40.663   4.918   44.650
H   38.747   6.208   45.271
H   37.837   4.875   45.619
H   41.860   -2.417   46.181
H   41.710   -2.208   47.885
H   42.802   -3.507   47.192
H   47.274   -2.349   48.384
H   47.463   -2.562   46.610
H   47.324   -3.998   47.736
H   49.270   0.798   46.330
H   49.287   3.812   46.988
H   50.149   0.501   48.350
H   48.691   1.209   48.982
H   50.078   2.247   48.569
H   48.791   3.839   44.260
H   49.522   2.321   44.499
H   51.685   3.446   44.413
H   51.117   4.183   45.897
H   50.923   4.944   44.375
H   45.704   8.893   46.099
H   46.506   7.818   44.888
H   47.194   7.988   46.489
H   40.941   6.741   46.049
H   39.211   8.596   49.473
H   39.282   6.883   50.030
H   37.929   7.480   49.022
H   38.026   6.642   40.933
H   37.532   4.863   40.975
H   40.263   4.591   40.816
H   39.165   7.300   38.118
H   37.740   6.292   38.718
H   38.468   5.883   37.247
H   41.903   5.178   38.915
H   41.031   5.651   37.480
H   41.057   3.500   36.760
H   40.136   3.076   38.104
H   41.967   1.869   38.944
H   42.918   3.376   39.009
H   42.658   2.015   36.279
H   43.744   0.030   37.285
H   44.166   0.932   38.782
H   45.008   1.074   37.260
H   44.741   3.794   37.529
H   43.390   4.106   36.288
H   44.473   2.486   34.861
H   45.888   2.534   35.945
H   46.592   4.383   34.965
H   45.079   5.199   35.245
H   44.896   3.207   32.946
H   47.156   5.052   32.396
H   46.279   3.802   31.457
H   47.171   3.478   32.992
H   45.036   6.007   32.047
H   44.117   5.993   33.466
H   42.305   5.358   32.238
H   43.013   3.793   31.695
H   44.179   5.816   30.070
H   42.488   5.924   29.964
H   43.724   3.165   29.980
H   42.016   2.688   28.369
H   41.215   4.248   28.854
H   41.531   2.959   29.993
H   44.905   3.264   28.046
H   44.914   4.933   27.987
H   43.654   4.064   27.193
