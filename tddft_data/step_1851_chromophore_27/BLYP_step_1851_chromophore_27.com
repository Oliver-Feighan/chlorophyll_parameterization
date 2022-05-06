%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1851_chromophore_27 TDDFT with blyp functional

0 1
Mg   -5.333   24.731   26.193
C   -3.779   26.836   28.516
C   -6.146   22.710   28.618
C   -6.627   22.798   23.743
C   -4.291   27.093   23.537
N   -4.886   24.758   28.325
C   -4.446   25.765   29.085
C   -4.728   25.529   30.538
C   -5.118   23.998   30.563
C   -5.427   23.723   29.141
C   -3.950   23.163   31.115
C   -5.912   26.388   31.061
C   -5.855   26.739   32.635
C   -6.919   26.241   33.539
O   -8.097   26.489   33.380
O   -6.404   25.512   34.598
N   -6.220   22.990   26.238
C   -6.501   22.304   27.320
C   -7.359   21.163   27.028
C   -7.412   21.126   25.562
C   -6.716   22.341   25.098
C   -7.820   20.149   28.002
C   -8.075   20.159   24.755
O   -8.185   20.268   23.575
C   -8.607   18.846   25.307
N   -5.379   24.886   23.939
C   -6.023   23.950   23.209
C   -6.055   24.280   21.735
C   -5.137   25.522   21.681
C   -4.876   25.900   23.137
C   -5.601   23.099   20.861
C   -5.685   26.679   20.857
C   -4.693   27.405   19.906
N   -4.207   26.520   25.954
C   -3.819   27.341   24.857
C   -3.003   28.498   25.378
C   -2.866   28.310   26.828
C   -3.652   27.126   27.093
C   -2.469   29.638   24.527
C   -2.354   28.816   28.099
O   -1.612   29.783   28.306
C   -2.884   27.871   29.119
C   -1.701   27.219   29.759
O   -0.908   26.425   29.204
O   -1.736   27.617   31.073
C   -0.708   26.894   31.921
C   -7.245   25.143   35.725
C   -8.025   23.951   35.267
C   -8.219   22.719   35.905
C   -7.562   22.297   37.170
C   -9.199   21.723   35.335
C   -10.673   21.909   35.829
C   -11.254   20.897   36.830
C   -12.441   20.164   36.153
C   -13.638   21.108   35.931
C   -12.708   18.852   36.981
C   -12.223   17.600   36.266
C   -11.510   16.618   37.161
C   -9.918   16.556   36.875
C   -9.152   17.709   37.465
C   -9.290   15.247   37.420
C   -7.790   15.014   36.994
C   -6.875   15.171   38.309
C   -5.518   14.462   37.958
C   -4.912   13.788   39.169
C   -4.471   15.341   37.279
H   -6.359   22.004   29.423
H   -7.028   22.173   22.943
H   -3.925   27.825   22.814
H   -3.852   25.704   31.163
H   -6.019   23.788   31.140
H   -3.110   23.785   31.425
H   -3.587   22.409   30.417
H   -4.256   22.735   32.070
H   -6.849   25.843   30.949
H   -5.968   27.273   30.428
H   -5.844   27.811   32.832
H   -4.886   26.451   33.043
H   -7.319   19.199   27.815
H   -8.879   19.929   27.869
H   -7.685   20.480   29.032
H   -9.060   18.206   24.549
H   -9.470   18.986   25.957
H   -7.919   18.252   25.907
H   -7.078   24.612   21.554
H   -4.135   25.351   21.286
H   -4.696   23.347   20.306
H   -6.424   22.642   20.312
H   -5.262   22.332   21.558
H   -6.106   27.406   21.552
H   -6.537   26.304   20.290
H   -3.698   27.011   19.699
H   -4.643   28.471   20.128
H   -5.261   27.330   18.979
H   -1.593   29.297   23.976
H   -2.276   30.538   25.111
H   -3.263   29.931   23.840
H   -3.549   28.390   29.810
H   0.227   27.403   31.687
H   -0.657   25.822   31.732
H   -0.864   26.990   32.996
H   -7.915   25.962   35.983
H   -6.637   24.878   36.590
H   -8.614   24.304   34.421
H   -8.267   22.144   37.987
H   -6.835   23.057   37.454
H   -7.002   21.370   37.042
H   -8.835   20.709   35.499
H   -9.276   21.820   34.252
H   -11.206   21.945   34.879
H   -10.666   22.888   36.307
H   -11.701   21.582   37.551
H   -10.363   20.340   37.121
H   -12.064   19.866   35.174
H   -13.819   21.607   36.884
H   -14.419   20.383   35.705
H   -13.454   21.883   35.187
H   -13.787   18.696   36.982
H   -12.388   18.993   38.013
H   -11.586   17.941   35.449
H   -13.038   17.098   35.744
H   -11.859   15.655   36.789
H   -11.654   16.697   38.238
H   -9.706   16.479   35.809
H   -9.925   18.335   37.911
H   -8.510   18.294   36.807
H   -8.577   17.447   38.353
H   -9.972   14.437   37.160
H   -9.424   15.413   38.489
H   -7.545   15.894   36.399
H   -7.597   14.077   36.472
H   -7.437   14.746   39.141
H   -6.737   16.203   38.631
H   -5.789   13.683   37.245
H   -4.945   12.725   38.929
H   -5.578   13.849   40.029
H   -3.881   14.028   39.431
H   -4.744   16.356   37.569
H   -4.314   15.050   36.241
H   -3.500   15.125   37.724

