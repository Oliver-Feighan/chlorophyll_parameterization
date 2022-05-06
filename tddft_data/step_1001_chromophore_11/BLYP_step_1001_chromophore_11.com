%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1001_chromophore_11 TDDFT with blyp functional

0 1
Mg   52.476   23.667   44.563
C   49.514   25.464   44.017
C   50.738   20.710   43.683
C   55.455   22.141   44.192
C   54.054   26.607   45.426
N   50.323   23.163   43.881
C   49.312   24.065   43.752
C   48.022   23.275   43.496
C   48.353   21.760   43.476
C   49.910   21.875   43.749
C   47.540   20.945   44.441
C   47.063   23.794   42.346
C   47.675   23.933   40.932
C   47.558   22.718   39.939
O   47.510   21.586   40.301
O   47.449   23.142   38.623
N   53.033   21.690   44.127
C   52.144   20.651   43.786
C   52.989   19.421   43.637
C   54.330   19.812   43.695
C   54.360   21.263   44.005
C   52.357   18.058   43.326
C   55.466   18.918   43.230
O   55.279   17.727   43.041
C   56.758   19.487   42.937
N   54.459   24.341   44.595
C   55.467   23.450   44.424
C   56.759   24.225   44.835
C   56.441   25.751   44.831
C   54.885   25.601   45.032
C   57.473   23.652   46.051
C   56.772   26.596   43.558
C   58.141   27.281   43.472
N   51.889   25.662   44.745
C   52.605   26.690   45.270
C   51.752   27.830   45.514
C   50.529   27.448   44.919
C   50.669   26.105   44.465
C   52.091   29.209   45.999
C   49.144   27.820   44.618
O   48.528   28.866   44.924
C   48.550   26.654   43.880
C   48.391   27.093   42.513
O   49.192   26.919   41.601
O   47.181   27.722   42.327
C   46.723   27.793   40.921
C   47.088   22.140   37.675
C   48.359   21.411   37.295
C   48.468   20.175   36.782
C   47.191   19.335   36.342
C   49.759   19.630   36.299
C   50.522   20.373   35.177
C   51.044   19.459   34.044
C   52.588   19.077   34.137
C   53.465   19.767   33.118
C   52.689   17.561   34.160
C   52.087   16.822   32.934
C   52.917   15.641   32.395
C   52.202   14.722   31.442
C   51.678   13.507   32.221
C   53.210   14.130   30.358
C   53.024   14.694   28.914
C   53.179   13.566   27.794
C   53.862   14.155   26.498
C   53.369   13.302   25.266
C   55.389   14.266   26.677
H   50.188   19.787   43.490
H   56.489   21.796   44.116
H   54.534   27.477   45.881
H   47.501   23.607   44.394
H   48.229   21.425   42.446
H   47.301   19.975   44.005
H   46.667   21.553   44.680
H   48.163   20.741   45.311
H   46.731   24.789   42.640
H   46.129   23.233   42.364
H   48.720   24.184   41.109
H   47.144   24.797   40.533
H   52.639   17.227   43.973
H   52.530   17.777   42.287
H   51.273   17.965   43.402
H   57.381   18.787   42.379
H   57.334   19.529   43.861
H   56.691   20.399   42.344
H   57.370   24.117   43.938
H   56.858   26.187   45.739
H   58.374   23.115   45.754
H   56.837   22.888   46.496
H   57.685   24.423   46.792
H   56.025   27.374   43.401
H   56.794   25.979   42.660
H   58.606   26.825   42.598
H   58.774   27.083   44.338
H   58.100   28.369   43.426
H   53.066   29.569   45.673
H   52.025   29.346   47.079
H   51.331   29.858   45.565
H   47.565   26.419   44.283
H   45.910   27.067   40.877
H   47.496   27.730   40.155
H   46.226   28.761   40.858
H   46.789   22.662   36.766
H   46.318   21.446   38.010
H   49.274   21.995   37.389
H   46.321   19.992   36.334
H   47.041   18.490   37.015
H   47.408   18.882   35.375
H   49.640   18.587   36.005
H   50.441   19.819   37.128
H   51.382   20.900   35.588
H   49.829   21.092   34.739
H   50.749   19.751   33.036
H   50.473   18.532   34.106
H   53.093   19.473   35.018
H   54.118   20.521   33.557
H   52.841   20.304   32.403
H   54.019   19.053   32.510
H   52.067   17.176   34.968
H   53.725   17.290   34.362
H   51.966   17.482   32.076
H   51.136   16.355   33.191
H   53.450   15.178   33.225
H   53.626   16.132   31.728
H   51.353   15.297   31.074
H   51.253   12.737   31.577
H   50.940   13.883   32.929
H   52.388   13.093   32.937
H   52.984   13.067   30.279
H   54.236   14.341   30.662
H   53.787   15.445   28.712
H   52.015   15.107   28.929
H   52.136   13.300   27.619
H   53.721   12.685   28.137
H   53.503   15.162   26.286
H   52.595   12.623   25.622
H   54.192   12.899   24.677
H   52.825   13.924   24.556
H   55.785   14.210   25.663
H   55.836   13.627   27.438
H   55.585   15.323   26.857

