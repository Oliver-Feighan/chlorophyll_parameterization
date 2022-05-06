%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1551_chromophore_9 TDDFT with wB97XD functional

0 1
Mg   35.791   0.690   29.920
C   33.458   1.839   32.278
C   38.130   0.672   32.482
C   38.091   -0.215   27.681
C   33.399   1.155   27.299
N   35.739   1.224   32.168
C   34.654   1.547   32.961
C   34.992   1.411   34.460
C   36.534   1.307   34.395
C   36.817   0.948   32.922
C   37.341   2.561   34.795
C   34.217   0.202   35.138
C   33.727   0.415   36.500
C   34.367   1.515   37.266
O   34.243   2.749   37.162
O   35.100   0.944   38.280
N   37.811   0.446   30.072
C   38.642   0.488   31.175
C   40.031   0.129   30.845
C   40.036   -0.236   29.540
C   38.530   -0.079   29.035
C   41.179   -0.064   31.782
C   41.205   -0.740   28.711
O   41.087   -1.075   27.550
C   42.672   -0.624   29.174
N   35.733   0.356   27.759
C   36.921   0.094   27.084
C   36.549   0.065   25.543
C   35.033   0.398   25.455
C   34.727   0.706   26.899
C   37.526   0.944   24.682
C   34.052   -0.626   24.744
C   33.694   -1.940   25.589
N   33.859   1.371   29.751
C   33.038   1.581   28.658
C   31.700   2.069   29.073
C   31.851   2.068   30.492
C   33.180   1.747   30.875
C   30.514   2.394   28.273
C   31.082   2.281   31.709
O   29.879   2.646   31.808
C   32.085   2.109   32.854
C   32.010   3.486   33.516
O   32.782   4.405   33.276
O   31.316   3.441   34.717
C   31.614   4.520   35.661
C   35.711   1.825   39.234
C   37.037   1.192   39.582
C   38.051   1.645   40.402
C   38.084   2.998   41.131
C   39.304   0.820   40.369
C   39.364   -0.414   41.323
C   40.458   -0.310   42.465
C   41.841   -0.848   41.990
C   41.814   -2.382   42.195
C   43.064   -0.155   42.674
C   43.858   0.782   41.724
C   45.308   0.983   42.216
C   45.849   2.328   41.734
C   47.389   2.370   41.796
C   45.298   3.486   42.646
C   44.741   4.856   41.930
C   44.646   6.101   42.877
C   45.032   7.419   42.163
C   46.532   7.549   42.079
C   44.324   8.616   42.662
H   38.788   0.612   33.351
H   39.012   -0.363   27.115
H   32.751   1.434   26.465
H   34.605   2.384   34.761
H   36.876   0.530   35.079
H   36.738   3.308   35.311
H   37.791   2.956   33.884
H   38.198   2.394   35.448
H   34.861   -0.677   35.116
H   33.308   0.074   34.550
H   33.774   -0.534   37.033
H   32.679   0.706   36.571
H   40.671   0.051   32.739
H   41.958   0.691   31.673
H   41.557   -1.083   31.705
H   42.992   0.311   29.632
H   43.330   -0.809   28.326
H   42.988   -1.416   29.854
H   36.715   -0.985   25.303
H   34.888   1.341   24.928
H   38.260   0.307   24.188
H   38.073   1.587   25.372
H   37.024   1.420   23.840
H   34.622   -1.045   23.915
H   33.130   -0.148   24.413
H   34.168   -1.933   26.571
H   33.879   -2.847   25.014
H   32.646   -1.973   25.885
H   30.594   2.165   27.210
H   30.336   3.455   28.450
H   29.706   1.790   28.684
H   31.790   1.458   33.677
H   30.930   4.488   36.510
H   31.625   5.535   35.263
H   32.611   4.385   36.080
H   35.101   1.784   40.136
H   35.942   2.815   38.840
H   37.151   0.209   39.126
H   37.102   3.456   41.014
H   38.865   3.594   40.658
H   38.197   2.868   42.207
H   40.121   1.538   40.441
H   39.392   0.416   39.361
H   39.712   -1.196   40.648
H   38.337   -0.595   41.642
H   40.226   -0.756   43.432
H   40.664   0.753   42.588
H   41.991   -0.653   40.928
H   41.318   -2.801   43.071
H   42.845   -2.571   42.492
H   41.403   -2.888   41.322
H   43.767   -0.956   42.900
H   42.749   0.445   43.528
H   43.351   1.744   41.653
H   43.879   0.251   40.772
H   45.931   0.175   41.831
H   45.361   0.996   43.304
H   45.575   2.519   40.696
H   47.777   3.342   42.099
H   47.835   2.305   40.804
H   47.964   1.618   42.336
H   46.105   3.760   43.325
H   44.406   3.109   43.147
H   43.723   4.671   41.587
H   45.305   5.059   41.020
H   45.395   5.877   43.637
H   43.636   6.143   43.284
H   44.725   7.308   41.123
H   47.213   6.698   42.095
H   46.799   8.260   42.862
H   46.833   8.106   41.192
H   45.015   9.360   43.059
H   43.632   8.454   43.488
H   43.847   9.010   41.765
