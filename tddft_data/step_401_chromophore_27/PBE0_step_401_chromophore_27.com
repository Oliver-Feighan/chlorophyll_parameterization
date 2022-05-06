%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_401_chromophore_27 TDDFT with PBE1PBE functional

0 1
Mg   -5.755   24.900   27.338
C   -4.098   26.839   29.764
C   -6.545   22.745   29.829
C   -7.237   22.971   24.969
C   -4.547   26.994   24.774
N   -5.349   24.960   29.564
C   -4.710   25.804   30.341
C   -4.758   25.415   31.848
C   -5.566   24.062   31.790
C   -5.888   23.946   30.289
C   -4.990   22.763   32.353
C   -5.279   26.514   32.847
C   -6.434   26.268   33.693
C   -6.190   25.354   34.857
O   -5.272   24.522   35.013
O   -7.095   25.502   35.881
N   -6.720   23.100   27.341
C   -6.905   22.356   28.493
C   -7.546   21.099   28.132
C   -7.743   21.146   26.664
C   -7.286   22.451   26.201
C   -7.907   19.978   29.200
C   -8.337   20.101   25.722
O   -8.520   20.249   24.525
C   -8.508   18.699   26.270
N   -5.802   24.998   25.117
C   -6.596   24.091   24.443
C   -6.874   24.631   23.026
C   -5.894   25.889   22.901
C   -5.346   25.955   24.320
C   -6.587   23.641   21.924
C   -6.642   27.240   22.417
C   -6.224   27.773   21.044
N   -4.472   26.546   27.172
C   -4.017   27.250   26.132
C   -3.135   28.256   26.568
C   -3.134   28.148   27.988
C   -3.912   27.064   28.344
C   -2.443   29.259   25.652
C   -2.686   28.721   29.236
O   -2.093   29.724   29.435
C   -3.222   27.867   30.412
C   -2.152   27.227   31.147
O   -1.463   26.254   30.862
O   -2.006   27.870   32.393
C   -0.804   27.488   33.181
C   -7.171   24.542   36.992
C   -8.198   23.425   36.663
C   -8.593   22.459   37.496
C   -8.109   22.326   39.010
C   -9.715   21.418   36.995
C   -11.149   21.730   37.545
C   -11.826   20.789   38.604
C   -12.713   19.663   38.054
C   -14.250   19.974   37.885
C   -12.503   18.334   38.659
C   -11.489   17.415   37.918
C   -10.729   16.608   38.909
C   -9.231   16.979   39.017
C   -9.088   18.201   39.869
C   -8.492   15.747   39.585
C   -7.250   15.457   38.652
C   -6.026   15.328   39.558
C   -5.024   14.261   39.155
C   -4.460   13.517   40.350
C   -3.987   14.798   38.092
H   -6.849   22.065   30.628
H   -7.759   22.372   24.219
H   -4.257   27.548   23.878
H   -3.737   25.156   32.126
H   -6.577   24.150   32.190
H   -5.782   22.264   32.912
H   -4.127   22.931   32.999
H   -4.576   22.083   31.609
H   -5.399   27.461   32.322
H   -4.487   26.632   33.587
H   -7.270   25.815   33.159
H   -6.830   27.240   33.987
H   -8.907   19.626   28.947
H   -8.044   20.388   30.201
H   -7.186   19.162   29.166
H   -7.534   18.356   26.618
H   -8.885   18.078   25.458
H   -9.378   18.647   26.924
H   -7.911   24.966   23.036
H   -5.110   25.643   22.185
H   -6.494   22.623   22.302
H   -5.899   23.893   21.117
H   -7.524   23.550   21.374
H   -6.324   28.028   23.100
H   -7.731   27.209   22.405
H   -5.799   26.948   20.473
H   -5.398   28.462   21.219
H   -7.096   28.078   20.465
H   -2.898   29.111   24.672
H   -1.406   29.008   25.432
H   -2.570   30.307   25.923
H   -3.781   28.491   31.110
H   -0.827   28.144   34.051
H   0.082   27.685   32.576
H   -0.839   26.412   33.352
H   -7.459   25.098   37.885
H   -6.229   24.056   37.249
H   -8.579   23.438   35.642
H   -7.357   21.538   39.069
H   -8.972   21.979   39.578
H   -7.614   23.234   39.352
H   -9.469   20.419   37.355
H   -9.679   21.395   35.906
H   -11.885   21.731   36.741
H   -11.048   22.770   37.857
H   -12.526   21.410   39.164
H   -11.203   20.366   39.392
H   -12.388   19.517   37.024
H   -14.567   21.015   37.944
H   -14.931   19.410   38.523
H   -14.500   19.646   36.876
H   -13.453   17.801   38.688
H   -12.323   18.374   39.733
H   -10.825   17.989   37.270
H   -12.073   16.731   37.302
H   -10.906   15.605   38.521
H   -11.307   16.505   39.827
H   -8.909   17.276   38.019
H   -8.675   19.074   39.364
H   -8.347   18.022   40.648
H   -10.029   18.442   40.363
H   -9.110   14.851   39.524
H   -8.188   15.888   40.623
H   -6.991   16.299   38.009
H   -7.320   14.518   38.103
H   -6.176   15.339   40.637
H   -5.537   16.280   39.349
H   -5.521   13.498   38.556
H   -3.378   13.580   40.466
H   -4.698   12.471   40.155
H   -4.773   13.982   41.285
H   -2.946   14.625   38.364
H   -3.957   15.879   37.959
H   -4.122   14.488   37.055

