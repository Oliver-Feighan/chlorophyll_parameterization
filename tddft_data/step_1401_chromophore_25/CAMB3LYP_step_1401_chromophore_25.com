%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1401_chromophore_25 TDDFT with cam-b3lyp functional

0 1
Mg   -2.852   34.936   26.406
C   -3.862   33.434   29.259
C   -1.199   37.274   28.428
C   -1.831   36.604   23.581
C   -4.731   32.804   24.596
N   -2.428   35.151   28.587
C   -2.955   34.413   29.568
C   -2.394   34.755   30.906
C   -1.642   36.128   30.581
C   -1.806   36.230   29.102
C   -2.256   37.387   31.336
C   -1.557   33.608   31.600
C   -1.955   33.102   32.972
C   -0.914   33.350   34.044
O   0.220   32.864   34.041
O   -1.500   34.143   35.063
N   -1.697   36.635   26.058
C   -1.100   37.454   27.001
C   -0.451   38.601   26.324
C   -0.562   38.416   24.911
C   -1.442   37.157   24.817
C   0.295   39.589   27.077
C   -0.052   39.194   23.749
O   -0.362   38.980   22.581
C   1.009   40.235   24.092
N   -3.295   34.810   24.375
C   -2.719   35.515   23.366
C   -3.109   34.904   21.951
C   -3.716   33.534   22.339
C   -3.954   33.703   23.853
C   -3.971   35.843   21.054
C   -2.754   32.396   22.087
C   -3.329   31.211   21.305
N   -4.105   33.391   26.817
C   -4.784   32.570   25.992
C   -5.322   31.476   26.758
C   -5.050   31.831   28.060
C   -4.206   32.952   28.009
C   -6.241   30.410   26.257
C   -5.232   31.411   29.397
O   -5.935   30.496   29.845
C   -4.430   32.458   30.224
C   -5.449   33.089   31.090
O   -6.434   33.757   30.716
O   -5.257   32.691   32.361
C   -6.039   33.359   33.356
C   -0.773   34.263   36.319
C   -1.519   34.951   37.371
C   -1.338   35.058   38.675
C   -0.070   34.437   39.329
C   -2.417   35.474   39.735
C   -2.360   36.934   40.307
C   -1.266   37.217   41.362
C   -1.735   37.717   42.748
C   -1.455   39.223   43.064
C   -1.305   36.836   43.919
C   0.197   36.844   44.246
C   0.579   36.883   45.792
C   1.383   35.503   46.149
C   1.021   35.180   47.610
C   2.955   35.664   45.975
C   3.635   34.395   45.345
C   3.710   34.479   43.811
C   5.061   35.141   43.346
C   4.997   35.968   42.034
C   6.224   34.117   43.277
H   -0.663   37.979   29.066
H   -1.506   36.972   22.606
H   -5.284   32.003   24.099
H   -3.248   34.938   31.558
H   -0.568   36.020   30.732
H   -1.349   37.986   31.410
H   -2.654   37.242   32.340
H   -2.945   37.952   30.707
H   -0.559   34.030   31.726
H   -1.485   32.782   30.892
H   -2.221   32.048   32.896
H   -2.859   33.635   33.266
H   1.300   39.419   26.690
H   0.139   39.533   28.154
H   0.062   40.626   26.836
H   1.364   40.489   23.094
H   1.869   39.794   24.596
H   0.695   41.085   24.698
H   -2.166   34.798   21.415
H   -4.552   33.191   21.730
H   -4.883   35.266   20.900
H   -3.440   35.995   20.115
H   -4.075   36.860   21.432
H   -2.294   32.113   23.034
H   -1.939   32.777   21.472
H   -3.315   30.408   22.042
H   -2.692   30.870   20.489
H   -4.381   31.266   21.023
H   -5.840   29.404   26.379
H   -6.393   30.568   25.189
H   -7.217   30.462   26.740
H   -3.758   31.902   30.877
H   -6.418   34.328   33.032
H   -5.319   33.508   34.160
H   -6.839   32.714   33.721
H   0.001   34.973   36.027
H   -0.251   33.358   36.629
H   -2.507   35.328   37.106
H   -0.408   33.621   39.967
H   0.369   35.188   39.986
H   0.791   34.113   38.744
H   -2.432   34.842   40.623
H   -3.375   35.465   39.214
H   -3.374   37.097   40.672
H   -2.143   37.631   39.497
H   -0.615   37.993   40.960
H   -0.627   36.366   41.596
H   -2.818   37.603   42.800
H   -0.882   39.328   43.985
H   -2.404   39.759   43.055
H   -0.749   39.732   42.408
H   -1.498   35.825   43.560
H   -1.959   36.994   44.776
H   0.733   37.664   43.769
H   0.673   35.913   43.939
H   -0.292   36.995   46.437
H   1.266   37.723   45.897
H   1.038   34.702   45.497
H   1.785   34.862   48.320
H   0.628   34.164   47.581
H   0.437   35.845   48.245
H   3.459   35.696   46.940
H   3.183   36.527   45.349
H   3.117   33.472   45.605
H   4.575   34.456   45.892
H   2.881   34.972   43.302
H   3.738   33.444   43.470
H   5.266   35.841   44.156
H   5.909   36.562   41.971
H   4.252   36.725   42.279
H   4.745   35.538   41.065
H   5.961   33.091   43.535
H   6.969   34.290   44.054
H   6.782   34.079   42.342

