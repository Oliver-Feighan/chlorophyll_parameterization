%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_251_chromophore_1 ZINDO

0 1
Mg   -2.044   17.491   26.738
C   -2.155   15.536   29.752
C   -2.484   20.324   28.824
C   -2.100   19.568   23.931
C   -2.029   14.735   24.850
N   -2.503   17.813   29.093
C   -2.371   16.884   30.064
C   -2.663   17.530   31.466
C   -2.966   19.011   31.040
C   -2.554   19.106   29.553
C   -4.458   19.415   31.307
C   -1.490   17.422   32.440
C   -1.829   17.031   33.841
C   -0.687   16.808   34.843
O   0.549   16.860   34.597
O   -1.198   16.647   36.136
N   -2.050   19.606   26.405
C   -2.247   20.565   27.393
C   -2.189   21.872   26.769
C   -1.982   21.711   25.409
C   -2.115   20.234   25.172
C   -2.467   23.169   27.473
C   -1.859   22.772   24.240
O   -1.680   22.558   23.066
C   -1.789   24.240   24.601
N   -2.213   17.239   24.736
C   -2.218   18.173   23.733
C   -2.507   17.544   22.350
C   -2.236   16.001   22.609
C   -2.120   15.997   24.181
C   -3.928   17.895   21.804
C   -0.961   15.441   21.908
C   0.371   15.966   22.439
N   -2.111   15.556   27.123
C   -2.028   14.458   26.222
C   -2.060   13.192   26.951
C   -2.054   13.576   28.304
C   -2.072   14.994   28.378
C   -2.028   11.859   26.361
C   -2.057   13.005   29.647
O   -1.985   11.838   29.994
C   -2.214   14.257   30.667
C   -1.015   13.993   31.581
O   0.195   14.138   31.330
O   -1.454   13.500   32.726
C   -0.507   13.417   33.911
C   -0.231   16.310   37.140
C   -0.334   17.397   38.195
C   -0.331   17.207   39.548
C   -0.296   15.961   40.347
C   -0.517   18.473   40.387
C   0.765   19.022   41.064
C   0.507   20.279   41.892
C   1.467   21.474   41.644
C   1.212   22.021   40.232
C   2.998   21.198   41.962
C   3.689   22.134   43.034
C   3.522   21.470   44.424
C   3.486   22.489   45.577
C   4.719   22.385   46.454
C   2.285   22.353   46.547
C   1.009   22.791   45.895
C   -0.034   21.634   45.838
C   -1.112   21.797   46.905
C   -1.053   20.699   47.978
C   -2.519   21.774   46.311
H   -2.545   21.193   29.482
H   -2.119   20.135   22.997
H   -1.948   13.880   24.176
H   -3.472   16.925   31.874
H   -2.322   19.720   31.562
H   -4.610   20.302   31.922
H   -5.094   18.586   31.618
H   -5.016   19.786   30.447
H   -1.292   18.475   32.641
H   -0.618   16.928   32.010
H   -2.497   16.188   34.025
H   -2.370   17.902   34.211
H   -2.541   23.028   28.551
H   -3.373   23.686   27.153
H   -1.693   23.931   27.380
H   -0.938   24.470   25.242
H   -2.765   24.395   25.061
H   -1.836   24.735   23.631
H   -1.820   18.066   21.683
H   -3.088   15.324   22.543
H   -3.894   18.556   20.938
H   -4.672   18.268   22.507
H   -4.290   16.905   21.526
H   -1.147   15.456   20.834
H   -0.925   14.371   22.112
H   0.308   16.763   23.179
H   0.886   16.379   21.572
H   0.967   15.255   23.010
H   -2.068   11.037   27.076
H   -1.096   11.824   25.797
H   -2.853   11.796   25.650
H   -3.078   14.151   31.323
H   -1.144   13.300   34.788
H   0.048   14.340   34.076
H   0.239   12.627   33.817
H   0.823   16.306   36.862
H   -0.393   15.316   37.556
H   -0.756   18.374   37.959
H   -1.326   15.740   40.629
H   0.404   16.208   41.145
H   0.048   15.057   39.844
H   -1.266   18.285   41.157
H   -1.002   19.296   39.863
H   1.543   19.228   40.329
H   1.090   18.154   41.637
H   0.615   19.948   42.925
H   -0.478   20.727   41.767
H   0.995   22.135   42.371
H   0.283   21.606   39.840
H   1.973   21.645   39.548
H   1.280   23.105   40.326
H   3.588   21.184   41.045
H   2.934   20.145   42.237
H   3.414   23.188   43.072
H   4.746   22.096   42.771
H   4.443   20.888   44.428
H   2.635   20.836   44.412
H   3.385   23.537   45.295
H   5.634   21.989   46.014
H   4.525   21.957   47.437
H   4.914   23.418   46.741
H   2.403   23.069   47.361
H   2.239   21.355   46.983
H   1.167   23.052   44.849
H   0.580   23.584   46.507
H   0.527   20.705   45.928
H   -0.516   21.620   44.861
H   -1.093   22.750   47.434
H   -0.807   19.774   47.456
H   -1.916   20.690   48.643
H   -0.142   20.922   48.533
H   -2.435   20.818   45.795
H   -2.753   22.626   45.672
H   -3.345   21.644   47.011

