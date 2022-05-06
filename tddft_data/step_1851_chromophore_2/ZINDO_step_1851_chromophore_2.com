%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1851_chromophore_2 ZINDO

0 1
Mg   1.907   -0.478   43.111
C   5.416   0.126   42.709
C   1.238   2.459   41.380
C   -1.311   -1.260   43.081
C   2.725   -3.635   44.501
N   3.215   1.109   42.107
C   4.613   1.137   42.103
C   5.199   2.411   41.480
C   3.799   3.290   41.443
C   2.639   2.214   41.614
C   3.683   4.432   42.475
C   6.026   2.293   40.142
C   5.378   1.477   39.077
C   5.436   2.005   37.642
O   5.689   3.159   37.281
O   4.965   0.988   36.851
N   0.220   0.391   42.247
C   0.116   1.633   41.655
C   -1.251   1.960   41.350
C   -1.952   0.848   41.887
C   -1.019   -0.039   42.501
C   -1.778   3.123   40.666
C   -3.438   0.729   41.745
O   -4.123   1.632   41.260
C   -4.238   -0.479   42.244
N   0.881   -2.154   43.729
C   -0.463   -2.217   43.755
C   -0.945   -3.571   44.252
C   0.254   -4.217   44.901
C   1.361   -3.331   44.247
C   -2.248   -3.697   45.094
C   0.413   -5.710   44.512
C   0.620   -6.702   45.672
N   3.662   -1.505   43.603
C   3.826   -2.796   44.109
C   5.220   -3.085   44.256
C   5.849   -1.952   43.784
C   4.909   -1.024   43.330
C   5.826   -4.325   44.811
C   7.187   -1.399   43.473
O   8.319   -1.812   43.733
C   6.987   0.014   42.817
C   7.647   1.082   43.583
O   8.255   1.984   43.073
O   7.634   0.796   44.940
C   8.634   1.668   45.680
C   4.898   1.276   35.370
C   5.344   0.091   34.569
C   4.601   -0.868   34.013
C   3.132   -1.101   34.100
C   5.378   -1.822   33.003
C   5.638   -1.387   31.581
C   5.680   0.170   31.373
C   5.430   0.667   29.979
C   6.701   0.812   29.127
C   4.546   1.914   29.824
C   3.072   1.554   29.423
C   2.279   1.111   30.702
C   1.066   2.046   30.949
C   0.932   2.368   32.453
C   -0.276   1.483   30.355
C   -1.266   2.676   30.118
C   -1.396   3.060   28.634
C   -2.787   3.276   28.106
C   -3.300   2.233   27.160
C   -2.865   4.516   27.289
H   1.058   3.433   40.919
H   -2.319   -1.670   43.180
H   2.894   -4.457   45.200
H   5.868   2.767   42.264
H   3.618   3.668   40.437
H   2.822   4.265   43.123
H   3.674   5.399   41.972
H   4.569   4.447   43.109
H   7.020   1.974   40.453
H   6.051   3.329   39.804
H   4.318   1.305   39.262
H   5.906   0.526   39.004
H   -0.973   3.672   40.177
H   -2.289   3.822   41.328
H   -2.427   2.923   39.814
H   -4.115   -0.412   43.325
H   -4.220   -1.439   41.728
H   -5.227   -0.125   41.954
H   -1.252   -4.002   43.299
H   0.350   -4.074   45.977
H   -2.043   -4.249   46.012
H   -3.193   -4.074   44.705
H   -2.473   -2.642   45.254
H   1.260   -5.673   43.827
H   -0.406   -6.082   43.896
H   -0.051   -7.559   45.611
H   0.188   -6.317   46.595
H   1.664   -6.972   45.832
H   5.694   -5.138   44.098
H   5.242   -4.467   45.720
H   6.900   -4.218   44.965
H   7.450   -0.067   41.833
H   9.517   1.876   45.075
H   8.902   1.016   46.511
H   8.248   2.622   46.038
H   5.525   2.093   35.014
H   3.864   1.499   35.106
H   6.381   0.118   34.237
H   2.935   -2.172   34.144
H   2.691   -0.612   33.232
H   2.687   -0.580   34.948
H   4.879   -2.791   33.037
H   6.255   -2.146   33.563
H   5.143   -1.983   30.815
H   6.675   -1.646   31.366
H   6.684   0.505   31.630
H   4.956   0.619   32.053
H   4.855   -0.151   29.544
H   6.722   1.438   28.235
H   6.895   -0.120   28.596
H   7.579   0.991   29.748
H   4.947   2.773   29.287
H   4.333   2.342   30.804
H   2.937   0.778   28.670
H   2.677   2.495   29.041
H   2.945   0.937   31.548
H   1.844   0.128   30.519
H   1.294   3.013   30.500
H   -0.081   2.330   32.854
H   1.270   3.386   32.650
H   1.640   1.736   32.989
H   -0.748   0.851   31.108
H   -0.170   0.821   29.496
H   -1.027   3.533   30.747
H   -2.151   2.238   30.581
H   -1.014   2.290   27.963
H   -0.799   3.972   28.632
H   -3.359   3.386   29.027
H   -4.079   2.613   26.500
H   -3.568   1.359   27.754
H   -2.440   1.911   26.573
H   -1.980   5.146   27.388
H   -3.790   4.990   27.616
H   -3.138   4.399   26.240

