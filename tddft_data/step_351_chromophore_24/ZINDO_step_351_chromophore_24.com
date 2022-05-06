%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_351_chromophore_24 ZINDO

0 1
Mg   -0.224   44.195   24.771
C   1.928   43.967   27.652
C   -2.794   43.366   26.795
C   -1.946   43.784   21.960
C   2.706   44.666   22.815
N   -0.356   43.462   27.036
C   0.566   43.655   27.961
C   0.016   43.418   29.373
C   -1.448   42.996   29.037
C   -1.610   43.306   27.533
C   -1.881   41.617   29.548
C   0.052   44.658   30.319
C   0.134   44.384   31.858
C   -0.185   42.928   32.343
O   0.515   42.004   32.055
O   -1.443   42.749   32.937
N   -2.123   43.762   24.384
C   -3.056   43.468   25.383
C   -4.420   43.327   24.690
C   -4.098   43.383   23.254
C   -2.679   43.643   23.132
C   -5.735   43.021   25.392
C   -4.968   43.084   22.071
O   -4.666   43.410   20.936
C   -6.360   42.569   22.287
N   0.249   44.331   22.742
C   -0.653   44.049   21.803
C   -0.018   44.203   20.399
C   1.418   44.739   20.648
C   1.430   44.591   22.166
C   -0.053   42.939   19.451
C   1.636   46.230   20.279
C   0.739   47.242   21.042
N   1.837   44.378   25.105
C   2.943   44.545   24.217
C   4.183   44.456   24.867
C   3.810   44.263   26.237
C   2.351   44.253   26.336
C   5.593   44.477   24.251
C   4.397   43.966   27.589
O   5.548   43.669   27.881
C   3.176   43.896   28.587
C   3.321   42.537   29.216
O   3.354   41.518   28.566
O   3.482   42.632   30.566
C   3.755   41.334   31.241
C   -1.579   41.452   33.583
C   -2.826   41.467   34.486
C   -2.898   41.092   35.789
C   -1.761   40.470   36.492
C   -4.310   40.893   36.517
C   -5.443   42.040   36.477
C   -6.672   41.661   35.715
C   -8.026   41.908   36.563
C   -8.408   43.382   36.776
C   -9.144   41.240   35.802
C   -9.466   39.837   36.381
C   -10.928   39.687   37.000
C   -10.972   39.580   38.523
C   -11.104   38.085   38.973
C   -12.065   40.551   39.168
C   -11.323   41.786   39.628
C   -12.250   43.030   39.868
C   -11.912   44.296   39.075
C   -12.156   45.628   39.803
C   -12.744   44.216   37.708
H   -3.679   43.270   27.427
H   -2.375   43.758   20.956
H   3.615   44.841   22.235
H   0.494   42.538   29.802
H   -2.051   43.635   29.682
H   -2.529   41.652   30.424
H   -0.988   41.081   29.867
H   -2.342   41.121   28.693
H   -0.811   45.316   30.220
H   0.857   45.328   30.016
H   -0.315   45.137   32.506
H   1.167   44.546   32.165
H   -6.137   42.047   25.112
H   -6.408   43.861   25.221
H   -5.614   42.916   26.470
H   -6.427   41.558   22.689
H   -6.962   42.449   21.386
H   -6.814   43.363   22.879
H   -0.661   44.874   19.830
H   2.189   44.114   20.197
H   -0.134   43.341   18.441
H   -0.881   42.233   19.515
H   0.819   42.306   19.612
H   1.184   46.423   19.306
H   2.693   46.450   20.425
H   1.271   47.607   21.920
H   -0.248   46.870   21.316
H   0.480   48.168   20.527
H   5.878   45.516   24.090
H   5.618   43.865   23.349
H   6.183   43.959   25.007
H   3.417   44.684   29.300
H   4.670   40.848   30.904
H   2.933   40.653   31.020
H   3.919   41.462   32.311
H   -0.617   41.159   34.003
H   -1.945   40.755   32.830
H   -3.686   41.739   33.874
H   -1.199   39.856   35.789
H   -1.941   39.843   37.365
H   -0.996   41.201   36.757
H   -4.082   40.645   37.554
H   -4.719   39.975   36.095
H   -4.971   42.911   36.022
H   -5.611   42.191   37.544
H   -6.700   40.597   35.481
H   -6.573   42.277   34.821
H   -7.852   41.395   37.509
H   -8.038   44.019   35.973
H   -8.101   43.787   37.741
H   -9.484   43.558   36.751
H   -8.796   41.085   34.780
H   -10.075   41.795   35.690
H   -8.679   39.455   37.032
H   -9.356   39.233   35.480
H   -11.359   38.764   36.612
H   -11.610   40.465   36.659
H   -10.023   39.880   38.966
H   -11.275   37.351   38.186
H   -11.935   37.880   39.647
H   -10.196   37.753   39.477
H   -12.443   40.136   40.102
H   -12.817   40.936   38.479
H   -10.510   42.007   38.937
H   -10.837   41.534   40.570
H   -12.143   43.281   40.923
H   -13.306   42.831   39.682
H   -10.863   44.267   38.782
H   -13.130   45.983   39.465
H   -11.523   46.495   39.611
H   -12.173   45.444   40.877
H   -13.378   43.339   37.584
H   -11.936   44.352   36.989
H   -13.449   45.025   37.515

