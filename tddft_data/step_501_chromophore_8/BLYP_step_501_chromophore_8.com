%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_501_chromophore_8 TDDFT with blyp functional

0 1
Mg   44.018   2.918   47.661
C   42.060   5.715   47.019
C   41.006   1.016   47.312
C   45.799   0.157   47.762
C   46.879   4.917   47.241
N   41.832   3.330   47.098
C   41.294   4.592   46.981
C   39.795   4.500   46.683
C   39.487   3.026   47.066
C   40.847   2.382   47.129
C   38.692   2.744   48.401
C   39.448   4.839   45.224
C   38.003   4.934   44.914
C   37.595   5.768   43.640
O   36.561   6.393   43.620
O   38.574   5.758   42.608
N   43.493   0.834   47.541
C   42.214   0.267   47.428
C   42.359   -1.162   47.434
C   43.776   -1.438   47.427
C   44.432   -0.097   47.608
C   41.156   -2.134   47.286
C   44.342   -2.859   47.341
O   43.627   -3.844   47.422
C   45.801   -3.041   47.163
N   46.062   2.629   47.741
C   46.538   1.382   47.824
C   48.115   1.403   47.696
C   48.510   2.913   47.470
C   47.020   3.561   47.426
C   48.934   0.687   48.802
C   49.246   3.077   46.123
C   50.259   4.227   46.068
N   44.474   4.920   47.221
C   45.652   5.613   47.207
C   45.347   7.044   47.090
C   43.980   7.117   47.109
C   43.479   5.835   47.120
C   46.366   8.188   47.106
C   42.874   8.023   47.028
O   42.816   9.227   46.979
C   41.577   7.148   46.981
C   40.791   7.445   48.210
O   41.191   7.462   49.341
O   39.437   7.571   47.854
C   38.526   7.781   48.943
C   38.071   6.418   41.375
C   39.089   6.260   40.305
C   39.318   7.078   39.267
C   38.626   8.382   38.897
C   40.522   6.690   38.405
C   40.470   5.398   37.428
C   41.373   4.297   37.994
C   41.754   3.324   36.790
C   41.626   1.870   37.238
C   43.203   3.523   36.403
C   43.287   3.601   34.862
C   43.848   4.897   34.344
C   44.335   4.780   32.912
C   45.894   4.742   32.773
C   43.812   5.895   32.018
C   42.755   5.417   30.999
C   42.976   6.002   29.572
C   43.474   4.948   28.519
C   42.352   4.127   27.890
C   44.281   5.686   27.419
H   40.026   0.536   47.285
H   46.516   -0.665   47.822
H   47.746   5.578   47.177
H   39.308   5.144   47.414
H   38.966   2.437   46.312
H   37.880   2.088   48.088
H   38.536   3.577   49.087
H   39.330   2.099   49.005
H   39.754   4.100   44.483
H   39.939   5.768   44.934
H   37.523   5.358   45.796
H   37.596   3.927   44.824
H   41.341   -2.416   46.249
H   40.169   -1.672   47.256
H   41.018   -3.020   47.906
H   46.059   -2.382   46.334
H   45.982   -4.066   46.839
H   46.212   -2.737   48.126
H   48.338   0.834   46.793
H   49.079   3.370   48.279
H   49.328   -0.239   48.382
H   48.300   0.568   49.680
H   49.825   1.270   49.035
H   48.453   3.223   45.390
H   49.732   2.168   45.768
H   51.186   3.720   45.798
H   50.324   4.819   46.981
H   50.021   4.901   45.245
H   46.220   8.977   46.369
H   47.304   7.651   46.963
H   46.441   8.506   48.146
H   41.039   7.449   46.082
H   38.182   8.812   48.860
H   38.863   7.606   49.965
H   37.694   7.091   48.805
H   37.828   7.459   41.589
H   37.147   5.930   41.063
H   39.748   5.400   40.426
H   37.733   8.490   39.512
H   38.241   8.371   37.878
H   39.238   9.275   39.025
H   41.355   6.687   39.107
H   40.783   7.520   37.748
H   40.956   5.674   36.493
H   39.460   5.042   37.223
H   40.721   3.838   38.737
H   42.236   4.674   38.544
H   40.926   3.507   36.106
H   40.604   1.510   37.355
H   42.031   1.915   38.249
H   42.104   1.149   36.574
H   43.829   2.679   36.693
H   43.613   4.403   36.899
H   42.272   3.527   34.472
H   43.873   2.735   34.554
H   44.606   5.076   35.107
H   42.995   5.574   34.358
H   43.961   3.856   32.470
H   46.192   4.956   31.747
H   46.206   3.732   33.037
H   46.360   5.452   33.456
H   44.626   6.309   31.423
H   43.578   6.751   32.650
H   41.844   5.856   31.405
H   42.705   4.330   30.930
H   43.524   6.942   29.641
H   42.033   6.437   29.243
H   44.181   4.323   29.062
H   41.691   3.784   28.686
H   42.642   3.268   27.284
H   41.658   4.810   27.401
H   44.669   4.981   26.684
H   45.261   6.003   27.775
H   43.663   6.462   26.968

