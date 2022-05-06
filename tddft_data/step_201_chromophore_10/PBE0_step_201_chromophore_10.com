%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_201_chromophore_10 TDDFT with PBE1PBE functional

0 1
Mg   40.914   7.930   28.804
C   42.865   9.294   31.249
C   38.841   6.570   31.222
C   39.179   6.418   26.352
C   43.228   8.858   26.483
N   40.995   7.901   30.958
C   41.779   8.655   31.782
C   41.231   8.545   33.323
C   40.014   7.600   33.184
C   39.902   7.330   31.674
C   40.093   6.190   33.993
C   40.907   10.009   33.806
C   39.841   10.144   34.908
C   40.294   10.974   36.212
O   40.248   12.229   36.309
O   40.790   10.124   37.183
N   39.201   6.653   28.804
C   38.475   6.220   29.869
C   37.396   5.459   29.418
C   37.353   5.592   27.969
C   38.647   6.287   27.594
C   36.470   4.677   30.364
C   36.226   5.104   26.947
O   36.272   5.302   25.750
C   34.964   4.591   27.468
N   41.116   7.704   26.682
C   40.305   7.063   25.842
C   40.812   7.065   24.456
C   42.056   8.081   24.440
C   42.159   8.241   25.965
C   41.132   5.605   24.056
C   41.692   9.468   23.790
C   40.599   10.346   24.484
N   42.707   8.810   28.806
C   43.610   9.089   27.823
C   44.857   9.608   28.391
C   44.570   9.791   29.768
C   43.287   9.281   29.930
C   46.155   9.949   27.579
C   45.031   10.333   30.996
O   46.054   10.932   31.315
C   43.986   9.982   32.083
C   44.556   9.293   33.326
O   44.675   8.090   33.529
O   44.565   10.264   34.321
C   44.930   9.837   35.615
C   41.714   10.717   38.131
C   40.770   11.032   39.305
C   41.122   11.445   40.545
C   42.420   11.982   40.967
C   40.089   11.675   41.689
C   39.421   10.390   42.179
C   37.830   10.446   42.027
C   37.030   10.215   43.369
C   37.296   8.726   43.861
C   35.436   10.419   43.149
C   34.693   11.124   44.346
C   34.413   12.635   43.988
C   33.075   13.028   44.735
C   32.249   14.066   43.922
C   33.392   13.620   46.065
C   33.567   12.498   47.128
C   34.962   12.620   47.780
C   35.143   11.531   48.897
C   35.902   12.016   50.162
C   35.800   10.280   48.347
H   38.068   6.285   31.937
H   38.507   6.078   25.562
H   43.916   9.215   25.714
H   42.104   8.184   33.867
H   39.074   8.011   33.553
H   40.556   5.483   33.306
H   39.099   5.839   34.271
H   40.731   6.282   34.872
H   40.551   10.710   33.050
H   41.802   10.412   34.280
H   39.558   9.207   35.389
H   38.962   10.673   34.541
H   36.719   4.656   31.425
H   36.417   3.632   30.058
H   35.525   5.198   30.209
H   34.354   4.405   26.584
H   34.628   5.441   28.063
H   35.090   3.707   28.093
H   40.038   7.459   23.799
H   43.033   7.739   24.098
H   40.904   4.949   24.897
H   42.140   5.265   23.820
H   40.583   5.454   23.126
H   41.331   9.286   22.777
H   42.623   10.034   23.766
H   39.830   10.497   23.726
H   41.081   11.215   24.930
H   40.016   9.877   25.276
H   45.840   10.507   26.698
H   46.443   8.937   27.295
H   46.935   10.503   28.101
H   43.506   10.938   32.288
H   44.110   10.002   36.314
H   45.756   10.453   35.971
H   45.155   8.774   35.707
H   42.170   11.671   37.867
H   42.487   10.034   38.483
H   39.760   10.668   39.116
H   43.116   11.696   40.178
H   42.823   11.609   41.908
H   42.280   13.060   40.882
H   39.335   12.400   41.385
H   40.599   12.141   42.533
H   39.543   10.374   43.262
H   39.749   9.481   41.674
H   37.544   9.643   41.348
H   37.502   11.345   41.505
H   37.448   10.903   44.104
H   36.383   8.383   44.347
H   38.095   8.865   44.590
H   37.679   8.182   42.998
H   34.921   9.461   43.073
H   35.306   10.929   42.195
H   35.216   11.101   45.302
H   33.780   10.552   44.512
H   34.412   12.880   42.926
H   35.224   13.141   44.512
H   32.415   12.163   44.802
H   31.264   13.639   43.731
H   32.748   14.245   42.970
H   31.994   14.969   44.477
H   32.553   14.250   46.361
H   34.255   14.282   46.138
H   33.497   11.499   46.698
H   32.782   12.654   47.868
H   35.178   13.597   48.213
H   35.651   12.449   46.954
H   34.115   11.329   49.199
H   35.508   11.562   51.071
H   35.755   13.086   50.311
H   36.981   11.922   50.034
H   36.525   10.539   47.575
H   35.000   9.783   47.798
H   36.304   9.640   49.072
