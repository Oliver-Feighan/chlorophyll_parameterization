%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_851_chromophore_7 ZINDO

0 1
Mg   25.680   -0.758   28.866
C   27.402   -0.834   31.946
C   22.712   -0.708   30.681
C   24.144   -0.751   25.990
C   28.781   -1.152   27.187
N   25.106   -0.665   31.106
C   25.977   -0.744   32.180
C   25.176   -0.624   33.500
C   23.725   -1.005   33.031
C   23.817   -0.706   31.540
C   23.435   -2.453   33.396
C   25.317   0.727   34.246
C   24.728   0.763   35.701
C   25.570   1.583   36.634
O   26.014   2.642   36.250
O   25.767   0.927   37.834
N   23.688   -0.530   28.395
C   22.644   -0.621   29.238
C   21.466   -0.528   28.468
C   21.832   -0.365   27.123
C   23.285   -0.594   27.087
C   20.065   -0.642   29.101
C   20.988   -0.037   25.871
O   21.385   0.053   24.696
C   19.592   0.263   26.148
N   26.355   -0.903   26.910
C   25.485   -0.956   25.847
C   26.188   -1.337   24.510
C   27.713   -1.139   24.852
C   27.617   -1.109   26.412
C   25.825   -2.767   23.876
C   28.391   0.149   24.311
C   29.940   0.058   24.231
N   27.694   -1.183   29.408
C   28.844   -1.183   28.667
C   29.981   -1.218   29.503
C   29.479   -1.054   30.797
C   28.075   -1.030   30.721
C   31.413   -1.480   29.033
C   29.902   -0.910   32.206
O   30.973   -0.853   32.800
C   28.523   -0.627   32.990
C   28.374   -1.542   34.076
O   28.400   -2.764   33.954
O   28.048   -0.905   35.210
C   28.028   -1.651   36.498
C   26.750   1.582   38.753
C   26.350   1.951   40.191
C   26.885   1.618   41.368
C   28.166   0.751   41.494
C   26.378   2.129   42.736
C   25.177   1.449   43.420
C   23.942   2.468   43.544
C   23.480   2.552   45.011
C   21.985   2.516   45.205
C   24.087   3.826   45.720
C   23.737   3.844   47.257
C   22.635   4.881   47.704
C   23.224   6.170   48.271
C   23.895   6.015   49.684
C   22.072   7.349   48.261
C   22.509   8.516   47.292
C   21.795   8.451   45.861
C   22.669   8.938   44.659
C   22.387   10.279   44.064
C   22.793   7.789   43.500
H   21.804   -0.740   31.288
H   23.604   -0.692   25.042
H   29.746   -0.937   26.723
H   25.537   -1.364   34.215
H   22.904   -0.515   33.554
H   22.701   -2.880   32.712
H   22.943   -2.484   34.368
H   24.344   -3.050   33.471
H   24.797   1.445   33.613
H   26.385   0.941   34.270
H   24.601   -0.276   36.006
H   23.757   1.257   35.702
H   19.957   -1.130   30.070
H   19.408   -1.318   28.553
H   19.632   0.348   29.239
H   19.109   0.665   25.257
H   19.607   0.961   26.985
H   19.008   -0.610   26.442
H   25.884   -0.561   23.807
H   28.198   -2.027   24.447
H   26.578   -3.484   24.204
H   25.753   -2.634   22.797
H   24.909   -3.236   24.235
H   28.005   1.009   24.857
H   28.028   0.356   23.304
H   30.341   0.193   23.226
H   30.317   -0.922   24.522
H   30.491   0.816   24.788
H   31.579   -0.858   28.153
H   31.528   -2.545   28.829
H   32.095   -1.056   29.770
H   28.611   0.402   33.338
H   27.145   -2.289   36.481
H   28.175   -1.084   37.417
H   28.909   -2.285   36.393
H   27.294   2.435   38.348
H   27.481   0.816   39.014
H   25.377   2.430   40.299
H   29.014   1.435   41.525
H   28.365   -0.042   40.773
H   28.248   0.216   42.441
H   26.318   3.209   42.597
H   27.184   2.020   43.462
H   25.511   1.009   44.360
H   24.849   0.627   42.784
H   23.149   2.134   42.875
H   24.129   3.502   43.253
H   23.855   1.739   45.633
H   21.883   1.970   46.143
H   21.505   2.029   44.357
H   21.742   3.576   45.278
H   23.647   4.757   45.363
H   25.140   4.091   45.625
H   24.624   3.886   47.889
H   23.319   2.860   47.470
H   21.954   4.459   48.443
H   22.005   5.088   46.839
H   24.042   6.403   47.590
H   23.666   5.042   50.121
H   23.627   6.790   50.401
H   24.960   6.024   49.454
H   22.093   7.818   49.245
H   21.133   6.880   47.963
H   23.568   8.541   47.037
H   22.179   9.462   47.720
H   20.948   9.137   45.833
H   21.468   7.439   45.623
H   23.673   9.165   45.019
H   22.499   10.282   42.980
H   23.051   11.067   44.420
H   21.360   10.531   44.330
H   23.774   7.814   43.026
H   22.055   7.991   42.724
H   22.718   6.830   44.013

