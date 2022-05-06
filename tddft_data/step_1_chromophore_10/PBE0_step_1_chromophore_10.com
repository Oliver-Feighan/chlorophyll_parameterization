%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1_chromophore_10 TDDFT with PBE1PBE functional

0 1
Mg   40.741   8.403   29.699
C   42.660   10.025   32.016
C   38.732   7.215   32.077
C   39.179   6.752   27.363
C   43.283   9.228   27.341
N   40.793   8.456   31.868
C   41.505   9.392   32.544
C   40.948   9.635   33.920
C   39.853   8.497   34.002
C   39.827   7.948   32.592
C   40.100   7.441   35.106
C   40.441   11.181   34.185
C   40.870   11.882   35.483
C   41.239   10.966   36.692
O   42.389   10.847   37.185
O   40.210   10.240   37.180
N   39.138   7.344   29.692
C   38.399   6.917   30.749
C   37.347   6.003   30.376
C   37.337   6.023   29.014
C   38.655   6.628   28.647
C   36.610   5.298   31.500
C   36.334   5.366   28.045
O   36.442   5.387   26.841
C   35.207   4.651   28.730
N   41.300   7.919   27.672
C   40.367   7.314   26.915
C   40.802   7.264   25.447
C   41.951   8.327   25.387
C   42.232   8.498   26.894
C   41.216   5.861   24.829
C   41.580   9.724   24.761
C   40.484   10.599   25.310
N   42.572   9.481   29.570
C   43.553   9.669   28.591
C   44.679   10.463   29.099
C   44.364   10.637   30.455
C   43.087   10.022   30.648
C   45.856   11.013   28.335
C   44.788   11.170   31.668
O   45.805   11.750   32.015
C   43.616   11.012   32.671
C   44.297   10.490   33.878
O   44.637   9.318   34.057
O   44.613   11.499   34.741
C   45.545   11.240   35.843
C   40.322   9.610   38.459
C   39.478   10.410   39.513
C   40.066   11.285   40.356
C   41.549   11.746   40.477
C   39.148   11.913   41.409
C   38.818   10.836   42.484
C   37.376   10.284   42.164
C   36.427   10.664   43.310
C   36.623   9.693   44.519
C   34.979   10.710   42.929
C   34.436   12.105   43.364
C   33.108   11.999   44.133
C   33.076   12.806   45.469
C   33.070   14.319   45.179
C   34.230   12.507   46.457
C   33.805   12.254   47.882
C   34.574   11.105   48.582
C   35.685   11.550   49.534
C   36.860   12.239   48.849
C   36.138   10.376   50.435
H   37.990   7.007   32.851
H   38.681   6.270   26.520
H   43.957   9.442   26.508
H   41.680   9.370   34.683
H   38.877   8.959   34.149
H   40.485   6.532   34.644
H   39.146   7.206   35.578
H   40.830   7.874   35.790
H   39.353   11.122   34.153
H   40.807   11.842   33.400
H   40.043   12.466   35.886
H   41.654   12.608   35.268
H   37.229   4.662   32.133
H   35.890   4.510   31.280
H   36.128   6.144   31.990
H   34.675   5.325   29.401
H   35.723   3.801   29.176
H   34.378   4.279   28.129
H   39.885   7.563   24.938
H   42.832   7.863   24.943
H   41.112   5.086   25.588
H   42.281   5.839   24.601
H   40.700   5.560   23.916
H   41.363   9.409   23.740
H   42.431   10.404   24.739
H   39.920   10.047   26.063
H   39.777   10.864   24.524
H   40.799   11.500   25.836
H   46.782   10.487   28.563
H   45.972   12.075   28.552
H   45.609   10.771   27.301
H   43.074   11.947   32.814
H   46.059   12.145   36.169
H   46.185   10.382   35.643
H   45.044   10.862   36.734
H   41.270   9.465   38.977
H   39.788   8.664   38.377
H   38.421   10.146   39.563
H   41.653   12.697   39.955
H   42.275   11.079   40.012
H   41.807   11.934   41.519
H   38.275   12.421   40.999
H   39.578   12.797   41.880
H   38.831   11.405   43.414
H   39.554   10.037   42.568
H   37.597   9.220   42.240
H   36.986   10.391   41.152
H   36.693   11.631   43.738
H   35.984   8.812   44.461
H   36.584   10.210   45.478
H   37.579   9.171   44.488
H   34.432   9.855   43.326
H   34.791   10.656   41.857
H   34.215   12.588   42.412
H   35.149   12.823   43.771
H   32.957   10.965   44.444
H   32.223   12.310   43.577
H   32.127   12.536   45.933
H   33.675   14.873   45.896
H   32.052   14.680   45.329
H   33.408   14.524   44.163
H   35.025   13.253   46.457
H   34.742   11.603   46.126
H   32.755   12.067   48.107
H   33.921   13.139   48.507
H   34.944   10.524   47.737
H   33.809   10.505   49.076
H   35.291   12.329   50.186
H   37.778   12.020   49.394
H   36.698   13.313   48.754
H   36.978   11.800   47.859
H   37.148   10.510   50.821
H   35.914   9.417   49.966
H   35.458   10.381   51.286

