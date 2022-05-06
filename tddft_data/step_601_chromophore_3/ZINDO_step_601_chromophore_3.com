%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_601_chromophore_3 ZINDO

0 1
Mg   1.804   8.203   26.292
C   2.364   10.318   28.917
C   2.922   5.421   28.410
C   1.646   6.150   23.607
C   1.636   10.948   24.197
N   2.544   7.928   28.366
C   2.485   8.916   29.330
C   2.650   8.332   30.739
C   3.041   6.896   30.516
C   2.919   6.756   29.000
C   4.356   6.417   31.149
C   1.426   8.623   31.666
C   1.627   8.688   33.200
C   2.864   7.906   33.846
O   3.982   8.354   34.112
O   2.431   6.664   34.270
N   2.191   6.115   26.004
C   2.578   5.158   27.019
C   2.632   3.878   26.379
C   2.222   3.948   25.016
C   1.938   5.401   24.823
C   3.152   2.639   27.103
C   2.020   2.910   23.992
O   1.701   3.140   22.815
C   2.179   1.418   24.324
N   1.671   8.539   24.227
C   1.644   7.546   23.325
C   1.669   8.056   21.852
C   1.398   9.586   22.128
C   1.615   9.725   23.615
C   2.982   7.746   21.006
C   -0.003   10.137   21.561
C   0.094   11.245   20.552
N   1.862   10.187   26.493
C   1.869   11.196   25.553
C   1.974   12.470   26.147
C   2.111   12.181   27.548
C   2.169   10.789   27.629
C   2.089   13.799   25.527
C   2.407   12.712   28.853
O   2.584   13.848   29.257
C   2.313   11.533   29.779
C   3.345   11.601   30.819
O   4.576   11.487   30.559
O   2.820   11.574   32.096
C   3.932   11.474   33.125
C   3.365   5.965   35.223
C   2.339   5.227   36.071
C   2.562   4.733   37.287
C   3.915   4.876   37.985
C   1.608   3.893   38.090
C   1.497   2.481   37.642
C   1.002   1.413   38.767
C   2.015   0.295   38.881
C   3.282   0.882   39.416
C   1.575   -0.908   39.717
C   2.261   -2.313   39.370
C   3.113   -2.750   40.574
C   4.519   -3.229   40.105
C   4.512   -4.773   39.861
C   5.599   -2.832   41.115
C   6.469   -1.645   40.498
C   7.319   -1.076   41.588
C   8.824   -1.161   41.273
C   9.533   -1.575   42.511
C   9.275   0.226   40.798
H   3.152   4.612   29.107
H   1.825   5.617   22.671
H   1.514   11.716   23.430
H   3.467   8.813   31.278
H   2.186   6.371   30.941
H   4.955   6.035   30.322
H   4.180   5.571   31.813
H   4.894   7.208   31.672
H   0.619   7.926   31.437
H   0.900   9.501   31.291
H   0.756   8.102   33.494
H   1.538   9.677   33.649
H   2.293   2.095   27.496
H   3.695   3.114   27.920
H   3.747   1.977   26.475
H   1.525   1.180   25.163
H   3.240   1.186   24.410
H   1.947   0.827   23.438
H   0.872   7.554   21.304
H   2.276   10.140   21.795
H   2.844   7.299   20.021
H   3.610   7.146   21.665
H   3.472   8.709   20.861
H   -0.529   10.570   22.412
H   -0.656   9.347   21.189
H   -0.595   10.995   19.745
H   1.111   11.517   20.268
H   -0.386   12.136   20.956
H   1.998   13.642   24.453
H   3.020   14.320   25.747
H   1.221   14.349   25.893
H   1.353   11.474   30.291
H   4.481   10.546   32.967
H   3.492   11.472   34.122
H   4.582   12.349   33.164
H   3.857   6.581   35.976
H   4.108   5.240   34.892
H   1.467   4.896   35.507
H   4.700   5.403   37.444
H   4.274   3.866   38.183
H   3.920   5.495   38.882
H   0.655   4.359   37.841
H   1.739   4.014   39.165
H   2.344   2.172   37.030
H   0.605   2.424   37.017
H   0.019   0.997   38.550
H   0.800   1.802   39.765
H   2.210   -0.166   37.913
H   3.492   0.481   40.408
H   3.261   1.968   39.500
H   4.182   0.707   38.825
H   0.489   -0.934   39.626
H   1.687   -0.799   40.796
H   2.883   -2.378   38.478
H   1.498   -3.088   39.307
H   2.567   -3.560   41.057
H   3.215   -1.966   41.324
H   4.738   -2.753   39.149
H   5.230   -5.218   40.550
H   4.724   -5.020   38.821
H   3.549   -5.227   40.094
H   6.408   -3.546   41.270
H   5.175   -2.520   42.069
H   5.783   -0.912   40.072
H   7.111   -2.016   39.700
H   7.025   -1.649   42.467
H   7.009   -0.069   41.869
H   9.091   -1.749   40.396
H   10.304   -2.281   42.203
H   8.889   -2.159   43.168
H   10.006   -0.779   43.085
H   10.280   0.183   40.378
H   9.323   0.892   41.659
H   8.580   0.615   40.053

