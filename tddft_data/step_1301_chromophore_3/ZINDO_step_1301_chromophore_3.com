%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1301_chromophore_3 ZINDO

0 1
Mg   2.566   8.685   26.391
C   2.974   10.979   29.163
C   3.032   6.173   28.716
C   2.862   6.671   23.797
C   2.834   11.413   24.250
N   3.062   8.618   28.624
C   3.078   9.668   29.555
C   3.108   9.115   30.984
C   3.075   7.641   30.771
C   3.076   7.434   29.282
C   4.151   6.870   31.508
C   1.857   9.630   31.644
C   1.427   9.071   32.999
C   2.488   8.439   33.851
O   3.672   8.744   33.892
O   1.866   7.407   34.581
N   2.786   6.652   26.288
C   2.929   5.789   27.374
C   2.922   4.459   26.778
C   2.882   4.511   25.307
C   2.818   6.024   25.068
C   3.107   3.175   27.517
C   2.840   3.348   24.279
O   2.617   3.552   23.052
C   3.140   1.941   24.666
N   2.601   9.015   24.229
C   2.792   7.968   23.420
C   2.645   8.408   21.981
C   2.849   9.987   22.062
C   2.831   10.195   23.591
C   3.481   7.637   20.909
C   1.748   10.870   21.391
C   0.338   10.781   21.811
N   2.702   10.794   26.553
C   2.816   11.756   25.609
C   2.966   13.081   26.223
C   2.964   12.797   27.622
C   2.875   11.409   27.765
C   3.167   14.234   25.454
C   3.175   13.433   28.901
O   3.360   14.651   29.121
C   3.235   12.282   29.977
C   4.548   12.371   30.553
O   5.637   12.517   29.985
O   4.440   12.262   31.910
C   5.636   12.169   32.717
C   2.844   6.580   35.340
C   2.127   5.697   36.330
C   2.310   5.436   37.650
C   3.553   5.816   38.489
C   1.295   4.558   38.354
C   1.209   3.113   37.796
C   1.170   2.025   38.884
C   2.352   1.050   38.889
C   3.538   1.725   39.603
C   1.863   -0.425   39.336
C   2.931   -1.539   39.065
C   3.158   -2.436   40.255
C   4.489   -3.350   40.112
C   4.073   -4.872   39.954
C   5.402   -3.082   41.361
C   6.661   -2.263   40.922
C   7.319   -1.705   42.233
C   8.833   -1.502   42.032
C   9.652   -2.108   43.220
C   9.156   0.065   41.807
H   2.945   5.365   29.445
H   2.991   5.991   22.952
H   3.014   12.264   23.590
H   3.997   9.487   31.495
H   2.089   7.322   31.107
H   4.674   6.404   30.672
H   3.515   6.295   32.181
H   4.899   7.467   32.028
H   1.048   9.359   30.967
H   2.015   10.707   31.700
H   0.647   8.315   32.905
H   0.999   9.894   33.572
H   4.034   2.652   27.280
H   2.296   2.476   27.314
H   3.290   3.333   28.580
H   2.902   1.306   23.813
H   2.385   1.619   25.383
H   4.180   1.818   24.969
H   1.586   8.256   21.772
H   3.857   10.233   21.729
H   4.342   7.082   21.282
H   3.831   8.328   20.142
H   2.878   6.918   20.353
H   1.738   10.615   20.331
H   2.013   11.927   21.425
H   -0.343   11.021   20.995
H   0.154   11.531   22.581
H   0.011   9.849   22.271
H   2.287   14.680   24.991
H   3.905   14.001   24.687
H   3.713   14.978   26.035
H   2.396   12.408   30.661
H   6.064   13.170   32.661
H   6.325   11.387   32.399
H   5.285   11.987   33.733
H   3.610   7.193   35.814
H   3.246   5.813   34.678
H   1.263   5.186   35.906
H   4.078   4.914   38.803
H   3.256   6.308   39.415
H   4.312   6.405   37.974
H   0.330   5.002   38.110
H   1.272   4.590   39.443
H   2.105   2.934   37.201
H   0.303   3.027   37.195
H   0.277   1.492   38.557
H   1.096   2.446   39.886
H   2.604   1.009   37.829
H   3.873   1.311   40.554
H   3.306   2.771   39.802
H   4.362   1.726   38.890
H   0.992   -0.525   38.688
H   1.560   -0.447   40.382
H   3.881   -1.108   38.748
H   2.614   -2.196   38.255
H   2.246   -2.998   40.455
H   3.279   -1.698   41.048
H   4.946   -3.102   39.154
H   3.073   -5.031   39.550
H   4.015   -5.460   40.870
H   4.712   -5.495   39.328
H   5.813   -4.028   41.714
H   4.833   -2.551   42.124
H   6.346   -1.424   40.301
H   7.510   -2.765   40.457
H   7.299   -2.469   43.010
H   6.788   -0.836   42.619
H   9.222   -1.902   41.096
H   10.357   -1.448   43.724
H   10.266   -2.934   42.860
H   9.100   -2.355   44.127
H   9.941   0.299   42.527
H   8.210   0.601   41.884
H   9.478   0.261   40.785

