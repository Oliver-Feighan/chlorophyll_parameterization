%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_401_chromophore_25 TDDFT with blyp functional

0 1
Mg   -3.105   34.520   26.515
C   -4.289   33.172   29.623
C   -1.212   36.697   28.431
C   -2.652   36.281   23.662
C   -4.841   32.085   24.880
N   -2.854   34.922   28.776
C   -3.293   34.147   29.847
C   -2.673   34.589   31.165
C   -1.912   35.863   30.809
C   -1.972   35.843   29.268
C   -2.416   37.171   31.481
C   -1.870   33.458   31.956
C   -2.458   33.128   33.393
C   -1.482   33.022   34.555
O   -0.646   32.125   34.640
O   -1.624   34.061   35.488
N   -2.085   36.298   26.139
C   -1.293   36.928   27.028
C   -0.560   37.927   26.214
C   -1.041   37.841   24.897
C   -1.928   36.705   24.820
C   0.390   38.862   26.886
C   -0.684   38.741   23.732
O   -0.849   38.373   22.568
C   -0.072   40.176   23.962
N   -3.743   34.382   24.571
C   -3.396   35.121   23.517
C   -3.938   34.487   22.183
C   -4.235   33.010   22.606
C   -4.263   33.154   24.128
C   -5.179   35.219   21.598
C   -3.229   31.920   22.140
C   -3.980   30.613   21.835
N   -4.431   33.015   27.087
C   -4.882   31.986   26.282
C   -5.559   30.970   27.165
C   -5.372   31.378   28.443
C   -4.703   32.636   28.381
C   -6.360   29.835   26.639
C   -5.565   31.069   29.880
O   -6.168   30.147   30.415
C   -4.883   32.214   30.626
C   -5.866   32.889   31.491
O   -6.633   33.780   31.168
O   -5.746   32.363   32.769
C   -6.483   32.923   33.809
C   -0.901   33.889   36.736
C   -1.550   34.614   37.831
C   -2.240   34.161   38.954
C   -2.796   32.841   39.036
C   -2.510   35.109   40.134
C   -1.382   35.332   41.101
C   -1.176   36.735   41.699
C   -1.914   36.913   43.070
C   -3.294   37.604   43.094
C   -0.984   37.697   44.036
C   -0.886   36.995   45.447
C   0.442   37.260   46.148
C   1.177   35.897   46.523
C   1.225   35.660   48.119
C   2.610   35.624   45.922
C   2.528   34.979   44.540
C   3.268   35.831   43.439
C   4.261   34.969   42.596
C   5.562   34.851   43.378
C   4.531   35.530   41.158
H   -0.559   37.432   28.907
H   -2.637   36.917   22.775
H   -5.292   31.144   24.558
H   -3.501   34.849   31.825
H   -0.890   35.779   31.178
H   -2.183   37.127   32.545
H   -3.503   37.183   31.400
H   -1.924   38.043   31.050
H   -0.861   33.844   32.102
H   -1.744   32.507   31.437
H   -2.969   32.166   33.374
H   -3.148   33.893   33.747
H   1.073   39.228   26.119
H   1.021   38.333   27.600
H   -0.137   39.598   27.494
H   -0.478   40.771   23.143
H   1.003   40.045   23.843
H   -0.264   40.554   24.966
H   -3.117   34.653   21.485
H   -5.290   32.803   22.432
H   -5.010   35.733   20.652
H   -5.474   36.049   22.241
H   -6.022   34.550   21.425
H   -2.583   31.752   23.002
H   -2.844   32.318   21.201
H   -3.393   30.069   21.095
H   -4.925   30.902   21.376
H   -4.100   29.964   22.703
H   -6.312   28.910   27.213
H   -6.150   29.641   25.587
H   -7.388   30.159   26.797
H   -4.036   31.844   31.204
H   -5.810   33.651   34.261
H   -6.797   32.221   34.581
H   -7.369   33.360   33.347
H   0.121   34.176   36.491
H   -0.849   32.829   36.984
H   -1.242   35.659   37.809
H   -2.218   32.234   38.340
H   -3.816   32.791   38.656
H   -2.807   32.394   40.030
H   -3.313   34.791   40.799
H   -2.876   36.053   39.729
H   -0.424   35.043   40.669
H   -1.445   34.517   41.821
H   -1.493   37.509   40.999
H   -0.088   36.787   41.729
H   -2.059   35.929   43.515
H   -4.002   37.100   42.437
H   -3.204   38.689   43.053
H   -3.507   37.425   44.148
H   -1.291   38.709   44.301
H   -0.019   37.919   43.580
H   -1.342   36.005   45.470
H   -1.642   37.535   46.017
H   0.272   37.825   47.064
H   1.116   37.853   45.530
H   0.633   35.020   46.174
H   0.486   34.913   48.407
H   0.833   36.537   48.635
H   2.162   35.370   48.595
H   3.072   34.883   46.575
H   3.301   36.466   45.962
H   1.494   34.971   44.196
H   2.810   33.929   44.460
H   3.824   36.628   43.934
H   2.587   36.315   42.740
H   3.832   33.969   42.545
H   5.815   33.793   43.434
H   5.435   35.219   44.396
H   6.285   35.383   42.759
H   3.789   35.046   40.522
H   5.503   35.275   40.735
H   4.526   36.617   41.230

