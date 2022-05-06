%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1101_chromophore_8 TDDFT with blyp functional

0 1
Mg   45.970   2.404   47.728
C   43.580   4.924   47.203
C   43.382   0.052   47.812
C   48.238   0.045   47.690
C   48.383   4.821   47.181
N   43.841   2.521   47.624
C   43.042   3.639   47.341
C   41.506   3.312   47.234
C   41.366   1.819   47.680
C   42.936   1.421   47.669
C   40.708   1.591   49.059
C   40.917   3.495   45.787
C   39.480   3.921   45.797
C   38.990   5.015   44.839
O   38.571   6.132   45.145
O   39.104   4.647   43.518
N   45.801   0.282   47.705
C   44.670   -0.484   47.798
C   45.017   -1.909   47.866
C   46.442   -1.959   47.758
C   46.934   -0.535   47.730
C   43.929   -2.931   47.992
C   47.245   -3.223   47.567
O   46.679   -4.353   47.615
C   48.749   -3.209   47.276
N   47.950   2.407   47.211
C   48.674   1.346   47.523
C   50.229   1.582   47.509
C   50.240   3.126   47.142
C   48.779   3.537   47.164
C   50.871   1.320   48.968
C   50.903   3.443   45.718
C   52.093   4.442   45.720
N   45.993   4.539   47.283
C   47.058   5.345   47.188
C   46.616   6.736   46.942
C   45.212   6.605   46.949
C   44.897   5.230   47.168
C   47.449   7.952   46.745
C   43.947   7.295   46.814
O   43.661   8.458   46.599
C   42.885   6.179   46.952
C   41.954   6.636   48.063
O   42.239   6.615   49.272
O   40.813   7.228   47.533
C   39.697   7.388   48.423
C   38.793   5.542   42.443
C   40.034   5.645   41.609
C   40.145   6.185   40.383
C   38.970   6.889   39.681
C   41.462   5.993   39.657
C   41.414   5.089   38.378
C   42.440   3.926   38.276
C   42.596   3.341   36.890
C   41.738   2.079   36.677
C   44.107   3.085   36.569
C   44.886   4.297   35.874
C   45.838   3.798   34.757
C   45.065   3.544   33.493
C   45.473   2.195   32.813
C   45.255   4.712   32.471
C   43.920   4.964   31.705
C   44.169   5.573   30.336
C   43.786   4.632   29.153
C   42.286   4.627   28.856
C   44.459   4.998   27.850
H   42.610   -0.706   47.961
H   49.003   -0.719   47.837
H   49.211   5.525   47.279
H   40.969   3.993   47.894
H   40.665   1.303   47.025
H   40.412   2.573   49.429
H   41.463   1.269   49.776
H   39.897   0.865   49.106
H   40.972   2.514   45.315
H   41.573   4.139   45.202
H   39.024   4.261   46.727
H   38.879   3.063   45.496
H   43.432   -3.075   47.033
H   43.237   -2.598   48.766
H   44.400   -3.875   48.269
H   49.409   -2.773   48.025
H   49.032   -2.876   46.278
H   49.102   -4.229   47.125
H   50.726   1.048   46.699
H   50.895   3.739   47.761
H   51.156   2.261   49.440
H   51.763   0.700   48.881
H   50.134   0.867   49.631
H   50.095   3.849   45.110
H   51.170   2.583   45.105
H   52.479   4.730   46.697
H   51.935   5.341   45.124
H   52.881   3.861   45.241
H   47.692   8.574   47.607
H   46.961   8.556   45.980
H   48.362   7.593   46.270
H   42.381   6.113   45.987
H   38.786   7.250   47.839
H   39.681   8.416   48.786
H   39.595   6.746   49.298
H   38.703   6.598   42.700
H   38.056   5.070   41.794
H   40.879   5.018   41.892
H   39.189   7.948   39.548
H   38.026   6.773   40.214
H   38.814   6.419   38.709
H   42.144   5.521   40.364
H   41.847   6.986   39.424
H   41.667   5.752   37.551
H   40.476   4.611   38.096
H   42.286   3.158   39.034
H   43.350   4.455   38.560
H   42.211   4.025   36.135
H   42.364   1.199   36.824
H   41.637   1.906   35.606
H   40.783   2.003   37.199
H   44.071   2.290   35.824
H   44.669   2.709   37.424
H   45.493   4.836   36.602
H   44.149   5.029   35.541
H   46.277   2.915   35.221
H   46.543   4.580   34.473
H   44.014   3.421   33.755
H   44.561   1.654   32.559
H   46.083   1.595   33.488
H   46.067   2.330   31.910
H   46.084   4.520   31.791
H   45.441   5.638   33.016
H   43.319   5.696   32.244
H   43.284   4.083   31.621
H   45.251   5.664   30.233
H   43.618   6.513   30.305
H   44.140   3.652   29.473
H   41.696   4.070   29.584
H   42.323   4.166   27.869
H   41.814   5.608   28.905
H   45.443   5.452   27.965
H   43.891   5.527   27.084
H   44.680   4.019   27.424
