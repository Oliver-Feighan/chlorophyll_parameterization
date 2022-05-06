%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_851_chromophore_16 TDDFT with blyp functional

0 1
Mg   40.821   41.578   27.534
C   40.026   43.766   30.188
C   41.778   39.216   29.867
C   42.182   39.759   25.037
C   40.589   44.310   25.334
N   41.099   41.608   29.770
C   40.502   42.503   30.640
C   40.720   42.030   32.127
C   41.201   40.520   31.964
C   41.461   40.448   30.438
C   42.354   40.100   32.799
C   39.395   42.255   32.935
C   39.244   41.429   34.238
C   38.636   42.092   35.426
O   37.475   42.318   35.611
O   39.624   42.437   36.287
N   41.729   39.673   27.456
C   41.864   38.814   28.481
C   42.277   37.524   27.951
C   42.464   37.647   26.532
C   42.197   39.128   26.294
C   42.356   36.290   28.849
C   42.819   36.612   25.505
O   43.086   36.852   24.344
C   42.769   35.137   25.955
N   41.280   42.048   25.506
C   41.831   41.084   24.745
C   41.798   41.421   23.257
C   41.323   42.931   23.316
C   40.988   43.127   24.785
C   43.222   41.090   22.525
C   40.034   43.194   22.447
C   38.647   42.978   23.140
N   40.303   43.628   27.677
C   40.308   44.625   26.705
C   39.840   45.834   27.309
C   39.605   45.512   28.649
C   39.994   44.177   28.825
C   39.500   47.212   26.685
C   39.222   46.081   29.910
O   38.730   47.167   30.180
C   39.493   44.967   31.074
C   40.519   45.534   31.965
O   41.655   45.784   31.704
O   39.923   45.722   33.189
C   40.595   46.648   34.044
C   39.200   43.057   37.578
C   38.710   41.948   38.458
C   38.050   42.160   39.593
C   37.788   43.476   40.256
C   37.571   40.954   40.382
C   38.495   40.521   41.515
C   38.719   38.907   41.451
C   38.754   38.300   42.850
C   40.166   37.753   43.202
C   37.721   37.240   43.072
C   36.991   37.211   44.457
C   36.493   35.767   44.811
C   35.158   35.709   45.559
C   35.246   36.580   46.844
C   34.778   34.285   45.852
C   33.497   33.871   45.017
C   32.126   33.972   45.693
C   31.310   35.244   45.338
C   30.344   34.922   44.191
C   30.356   35.736   46.499
H   41.887   38.421   30.608
H   42.496   39.140   24.194
H   40.398   45.184   24.708
H   41.455   42.661   32.627
H   40.349   39.889   32.215
H   43.070   39.536   32.202
H   42.040   39.459   33.623
H   42.951   40.872   33.286
H   38.647   41.882   32.235
H   39.239   43.305   33.182
H   40.243   41.072   34.491
H   38.698   40.486   34.198
H   42.224   36.352   29.929
H   43.269   35.742   28.619
H   41.527   35.652   28.542
H   43.594   35.076   26.665
H   43.092   34.456   25.168
H   41.804   34.845   26.370
H   41.034   40.836   22.744
H   42.141   43.561   22.965
H   43.484   42.007   21.997
H   43.074   40.266   21.827
H   44.073   40.852   23.162
H   40.180   42.501   21.618
H   40.141   44.234   22.138
H   38.863   42.351   24.006
H   37.984   42.379   22.515
H   38.189   43.947   23.338
H   40.092   48.058   27.035
H   38.461   47.338   26.989
H   39.425   47.173   25.598
H   38.499   44.910   31.518
H   40.930   46.118   34.936
H   39.815   47.369   34.288
H   41.497   47.135   33.673
H   38.328   43.691   37.416
H   40.060   43.612   37.955
H   38.790   41.011   37.907
H   36.726   43.467   40.501
H   37.938   44.305   39.564
H   38.289   43.679   41.203
H   37.407   40.203   39.610
H   36.643   41.313   40.826
H   37.971   40.671   42.459
H   39.442   41.058   41.471
H   39.687   38.795   40.962
H   38.029   38.329   40.837
H   38.548   39.088   43.574
H   40.100   37.091   44.065
H   40.652   38.724   43.286
H   40.530   37.240   42.311
H   38.223   36.288   42.896
H   36.907   37.509   42.399
H   36.218   37.980   44.454
H   37.692   37.565   45.213
H   37.165   35.245   45.491
H   36.299   35.208   43.895
H   34.436   36.192   44.901
H   34.930   37.607   46.663
H   36.279   36.802   47.112
H   34.763   36.155   47.724
H   34.507   34.151   46.900
H   35.564   33.567   45.618
H   33.590   32.804   44.814
H   33.506   34.349   44.038
H   32.325   34.003   46.764
H   31.502   33.078   45.659
H   32.041   36.037   45.179
H   30.364   33.888   43.846
H   30.637   35.618   43.404
H   29.306   35.182   44.394
H   29.316   35.945   46.249
H   30.726   36.721   46.784
H   30.484   35.054   47.340
