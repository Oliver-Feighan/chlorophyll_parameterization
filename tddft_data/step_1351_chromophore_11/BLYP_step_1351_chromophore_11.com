%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1351_chromophore_11 TDDFT with blyp functional

0 1
Mg   52.047   24.352   45.213
C   49.278   26.447   44.225
C   50.256   21.701   44.030
C   54.848   22.540   45.298
C   53.810   27.352   45.845
N   50.018   24.064   44.240
C   49.048   25.081   43.905
C   47.765   24.386   43.396
C   48.018   22.876   43.804
C   49.521   22.873   44.002
C   47.348   22.479   45.136
C   47.478   24.703   41.912
C   48.561   24.949   40.940
C   48.267   25.263   39.468
O   48.278   26.393   39.019
O   48.039   24.112   38.804
N   52.556   22.354   44.745
C   51.608   21.436   44.334
C   52.211   20.180   44.278
C   53.573   20.372   44.555
C   53.671   21.830   44.868
C   51.478   18.838   43.972
C   54.605   19.282   44.727
O   54.252   18.097   44.568
C   55.986   19.589   45.148
N   54.021   24.947   45.317
C   54.993   23.963   45.551
C   56.343   24.554   46.101
C   56.045   26.066   46.064
C   54.497   26.145   45.776
C   56.836   23.829   47.383
C   56.649   26.837   44.805
C   57.045   28.293   45.174
N   51.644   26.450   45.238
C   52.432   27.469   45.663
C   51.657   28.734   45.567
C   50.411   28.314   45.070
C   50.463   26.935   44.794
C   52.059   30.093   45.961
C   49.070   28.804   44.895
O   48.493   29.820   45.185
C   48.277   27.659   44.220
C   47.826   28.164   42.921
O   48.558   28.475   41.996
O   46.465   28.307   42.853
C   45.905   28.679   41.544
C   47.500   24.174   37.448
C   48.452   23.376   36.610
C   48.517   22.035   36.482
C   47.727   20.991   37.188
C   49.604   21.539   35.504
C   49.840   20.064   35.201
C   51.350   19.714   34.704
C   52.027   18.627   35.627
C   53.565   18.935   35.840
C   51.736   17.180   35.133
C   52.465   16.660   33.858
C   51.451   16.446   32.694
C   51.398   14.953   32.233
C   50.843   14.030   33.312
C   52.710   14.387   31.543
C   53.016   14.977   30.077
C   53.011   13.825   28.973
C   53.513   14.363   27.621
C   52.691   13.582   26.508
C   55.037   14.086   27.508
H   49.736   20.785   43.743
H   55.790   22.052   45.556
H   54.306   28.230   46.263
H   46.970   24.758   44.042
H   47.762   22.257   42.944
H   47.533   23.256   45.877
H   47.650   21.505   45.521
H   46.301   22.299   44.892
H   46.787   25.546   41.905
H   46.827   23.938   41.488
H   49.238   24.096   40.989
H   49.036   25.871   41.275
H   51.375   18.633   42.906
H   50.425   18.809   44.252
H   51.915   17.940   44.409
H   56.486   18.621   45.129
H   55.960   19.980   46.165
H   56.525   20.259   44.478
H   57.003   24.242   45.291
H   56.201   26.581   47.012
H   56.926   24.619   48.128
H   57.812   23.479   47.047
H   56.231   23.004   47.759
H   55.920   26.776   43.997
H   57.510   26.260   44.468
H   56.546   28.866   44.392
H   58.133   28.292   45.108
H   56.719   28.627   46.158
H   51.400   30.546   46.702
H   52.132   30.707   45.064
H   53.069   30.166   46.365
H   47.391   27.570   44.849
H   46.069   27.779   40.952
H   46.353   29.567   41.097
H   44.832   28.791   41.702
H   47.686   25.186   37.087
H   46.484   23.814   37.283
H   49.195   23.916   36.024
H   47.035   20.507   36.499
H   47.167   21.383   38.036
H   48.431   20.308   37.665
H   50.521   21.950   35.927
H   49.443   21.999   34.529
H   49.303   19.712   34.320
H   49.560   19.454   36.059
H   51.927   20.637   34.749
H   51.314   19.328   33.685
H   51.568   18.769   36.606
H   53.849   19.520   36.715
H   53.876   19.388   34.899
H   54.212   18.063   35.743
H   50.647   17.122   35.158
H   52.124   16.513   35.902
H   53.035   15.747   34.032
H   53.160   17.406   33.475
H   51.731   17.037   31.822
H   50.439   16.582   33.074
H   50.703   14.955   31.393
H   49.913   13.510   33.085
H   50.533   14.557   34.215
H   51.679   13.404   33.623
H   52.570   13.319   31.373
H   53.479   14.601   32.285
H   53.916   15.592   30.081
H   52.136   15.567   29.825
H   52.027   13.364   28.894
H   53.632   12.986   29.286
H   53.488   15.446   27.504
H   52.064   12.796   26.929
H   53.383   13.121   25.803
H   51.993   14.232   25.980
H   55.585   14.488   28.360
H   55.372   14.498   26.556
H   55.226   13.024   27.352
