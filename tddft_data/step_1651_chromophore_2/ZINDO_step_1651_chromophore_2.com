%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1651_chromophore_2 ZINDO

0 1
Mg   3.046   0.322   44.136
C   6.430   1.073   43.734
C   2.249   3.424   42.961
C   -0.039   -0.821   43.548
C   4.086   -3.019   44.863
N   4.199   2.031   43.313
C   5.571   2.176   43.393
C   5.985   3.555   42.824
C   4.688   4.415   42.985
C   3.637   3.305   43.093
C   4.674   5.484   44.166
C   6.656   3.529   41.391
C   6.325   2.374   40.469
C   5.571   2.645   39.221
O   4.540   3.274   39.139
O   6.054   1.850   38.186
N   1.263   1.154   43.574
C   1.151   2.465   43.132
C   -0.234   2.793   42.898
C   -0.885   1.531   42.944
C   0.063   0.546   43.353
C   -0.788   4.167   42.760
C   -2.340   1.291   42.680
O   -3.075   2.213   42.281
C   -2.994   -0.073   42.741
N   2.168   -1.681   44.116
C   0.857   -1.830   43.919
C   0.428   -3.297   43.937
C   1.736   -4.017   44.484
C   2.775   -2.852   44.497
C   -0.766   -3.633   44.854
C   2.250   -5.177   43.539
C   2.452   -6.556   44.148
N   4.822   -0.812   44.290
C   5.057   -2.088   44.756
C   6.463   -2.237   44.962
C   7.078   -1.128   44.431
C   6.043   -0.270   44.141
C   7.177   -3.427   45.593
C   8.276   -0.462   44.126
O   9.427   -0.956   44.276
C   7.902   1.029   43.616
C   8.735   2.012   44.411
O   9.384   2.878   43.857
O   8.584   1.895   45.797
C   9.335   2.893   46.538
C   5.083   1.774   37.030
C   5.777   0.997   35.904
C   5.342   -0.114   35.299
C   4.036   -0.869   35.599
C   6.278   -0.739   34.224
C   5.717   -0.994   32.833
C   6.202   -0.056   31.678
C   4.993   0.596   30.970
C   5.459   1.102   29.570
C   4.173   1.641   31.694
C   2.670   1.519   31.343
C   1.906   0.980   32.454
C   1.296   2.073   33.441
C   1.999   2.103   34.769
C   -0.245   1.969   33.539
C   -1.016   3.078   32.892
C   -2.443   2.585   32.525
C   -2.902   2.826   31.084
C   -3.361   4.278   30.746
C   -3.835   1.677   30.612
H   1.801   4.323   42.533
H   -1.025   -1.278   43.435
H   4.345   -3.997   45.275
H   6.847   3.898   43.397
H   4.437   4.942   42.064
H   5.529   5.410   44.837
H   3.751   5.403   44.741
H   4.666   6.505   43.785
H   7.717   3.759   41.479
H   6.234   4.389   40.871
H   5.618   1.700   40.953
H   7.317   2.047   40.159
H   -1.822   4.254   42.427
H   -0.270   4.754   42.001
H   -0.639   4.705   43.696
H   -2.773   -0.515   43.712
H   -2.466   -0.714   42.035
H   -4.030   -0.035   42.403
H   0.198   -3.545   42.900
H   1.682   -4.364   45.516
H   -1.058   -2.676   45.287
H   -0.558   -4.354   45.644
H   -1.626   -4.003   44.296
H   3.225   -4.974   43.096
H   1.437   -5.142   42.815
H   3.348   -6.959   43.676
H   1.729   -7.327   43.884
H   2.556   -6.628   45.230
H   7.021   -4.271   44.921
H   6.798   -3.755   46.561
H   8.235   -3.199   45.720
H   8.229   1.215   42.593
H   10.374   2.920   46.212
H   9.370   2.661   47.602
H   8.921   3.885   46.358
H   4.792   2.713   36.560
H   4.140   1.368   37.395
H   6.621   1.490   35.420
H   3.360   -0.940   34.747
H   3.544   -0.475   36.489
H   4.405   -1.879   35.781
H   6.584   -1.766   34.421
H   7.232   -0.214   34.172
H   4.627   -1.033   32.835
H   5.989   -2.030   32.632
H   6.857   -0.595   30.995
H   6.716   0.804   32.108
H   4.362   -0.255   30.710
H   5.061   2.087   29.324
H   5.114   0.467   28.753
H   6.547   1.087   29.515
H   4.494   2.643   31.411
H   4.238   1.665   32.782
H   2.500   0.872   30.482
H   2.329   2.489   30.982
H   2.482   0.272   33.050
H   1.072   0.365   32.114
H   1.613   3.016   32.994
H   1.988   1.155   35.306
H   1.495   2.793   35.446
H   3.019   2.480   34.694
H   -0.597   1.770   34.551
H   -0.557   0.992   33.170
H   -0.518   3.593   32.070
H   -1.137   3.841   33.661
H   -3.099   3.060   33.255
H   -2.488   1.530   32.793
H   -1.995   2.737   30.487
H   -2.847   4.678   29.872
H   -3.383   4.895   31.644
H   -4.406   4.157   30.461
H   -4.546   1.395   31.389
H   -3.170   0.815   30.569
H   -4.331   1.883   29.663

