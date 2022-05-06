%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1451_chromophore_13 TDDFT with PBE1PBE functional

0 1
Mg   46.810   25.076   28.883
C   47.256   27.418   31.445
C   45.879   22.770   31.279
C   46.696   22.669   26.497
C   47.786   27.414   26.532
N   46.621   25.139   31.212
C   46.866   26.175   31.971
C   46.434   25.778   33.408
C   46.156   24.260   33.407
C   46.223   24.036   31.891
C   47.163   23.311   34.183
C   45.293   26.650   33.964
C   45.595   27.657   35.125
C   44.573   27.544   36.242
O   43.364   27.522   36.069
O   45.169   27.538   37.480
N   46.192   23.029   28.835
C   45.879   22.328   29.964
C   45.471   21.006   29.491
C   45.644   20.888   28.057
C   46.186   22.246   27.706
C   45.185   19.852   30.520
C   45.413   19.648   27.091
O   45.683   19.629   25.870
C   45.098   18.413   27.705
N   47.370   25.084   26.796
C   47.233   23.896   26.097
C   47.620   24.195   24.587
C   47.525   25.776   24.542
C   47.575   26.136   26.022
C   48.922   23.448   24.181
C   46.212   26.330   23.933
C   44.808   26.053   24.672
N   47.347   26.986   28.941
C   47.682   27.890   27.910
C   47.941   29.134   28.493
C   47.858   28.988   29.848
C   47.459   27.696   30.089
C   48.475   30.250   27.771
C   48.085   29.634   31.147
O   48.590   30.718   31.346
C   47.700   28.681   32.245
C   48.931   28.557   33.105
O   50.044   28.069   32.902
O   48.644   29.072   34.350
C   49.682   28.820   35.408
C   44.341   27.427   38.669
C   44.311   26.113   39.280
C   43.978   25.820   40.538
C   43.406   26.869   41.567
C   44.235   24.480   41.051
C   43.062   23.478   40.829
C   42.626   22.917   42.222
C   41.131   22.629   42.311
C   40.773   21.555   41.198
C   40.213   23.860   42.405
C   39.407   23.875   43.819
C   39.495   25.150   44.629
C   39.663   24.886   46.154
C   39.138   25.932   47.057
C   41.172   24.531   46.347
C   41.421   23.286   47.264
C   42.752   22.644   46.744
C   44.077   23.427   46.951
C   44.494   24.281   45.727
C   45.111   22.610   47.621
H   45.506   22.112   32.067
H   46.751   21.872   25.751
H   47.957   28.166   25.759
H   47.337   25.855   34.014
H   45.210   24.002   33.884
H   47.345   23.756   35.161
H   48.084   23.151   33.622
H   46.674   22.368   34.427
H   44.555   26.013   34.451
H   44.843   27.122   33.090
H   45.546   28.619   34.615
H   46.576   27.542   35.587
H   44.156   19.536   30.349
H   45.244   20.098   31.581
H   45.805   18.985   30.296
H   44.301   18.381   28.448
H   45.938   18.243   28.379
H   45.126   17.637   26.941
H   46.764   23.781   24.054
H   48.386   26.228   24.049
H   49.493   24.215   23.657
H   48.724   22.606   23.519
H   49.510   23.091   25.027
H   46.230   26.293   22.844
H   46.360   27.408   24.000
H   44.239   25.624   23.847
H   44.344   26.939   25.106
H   44.816   25.195   25.344
H   47.952   31.197   27.899
H   48.667   30.014   26.724
H   49.506   30.400   28.092
H   46.903   29.215   32.763
H   49.348   28.267   36.287
H   49.895   29.796   35.842
H   50.673   28.474   35.114
H   43.321   27.693   38.393
H   44.689   28.245   39.299
H   44.826   25.352   38.693
H   42.683   26.359   42.204
H   42.885   27.680   41.059
H   44.226   27.203   42.203
H   44.568   24.472   42.089
H   45.121   24.171   40.495
H   43.410   22.762   40.085
H   42.244   23.992   40.324
H   42.757   23.540   43.106
H   43.234   22.048   42.475
H   40.927   22.124   43.255
H   39.984   21.916   40.538
H   40.245   20.776   41.747
H   41.600   21.073   40.676
H   39.562   23.913   41.532
H   40.872   24.723   42.503
H   39.607   23.041   44.491
H   38.367   23.776   43.510
H   38.555   25.699   44.572
H   40.285   25.791   44.236
H   39.007   24.087   46.501
H   39.894   26.220   47.787
H   38.353   25.486   47.668
H   38.746   26.783   46.500
H   41.685   25.404   46.751
H   41.677   24.437   45.385
H   40.623   22.548   47.347
H   41.494   23.704   48.268
H   42.666   22.341   45.700
H   42.773   21.775   47.401
H   43.767   24.185   47.671
H   44.404   25.352   45.913
H   43.865   24.041   44.870
H   45.540   24.017   45.573
H   44.874   21.552   47.510
H   45.102   22.900   48.672
H   46.125   22.822   47.280

