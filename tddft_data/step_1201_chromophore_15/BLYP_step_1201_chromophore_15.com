%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1201_chromophore_15 TDDFT with blyp functional

0 1
Mg   46.137   35.499   28.073
C   44.737   33.236   30.475
C   46.672   37.721   30.514
C   47.616   37.317   25.763
C   45.451   32.980   25.670
N   45.669   35.461   30.301
C   45.256   34.424   31.070
C   45.436   34.795   32.538
C   45.559   36.365   32.467
C   45.994   36.607   30.982
C   44.223   37.158   32.668
C   46.736   34.179   33.111
C   46.912   34.020   34.626
C   46.085   34.795   35.602
O   44.852   34.707   35.657
O   46.865   35.831   36.106
N   46.883   37.437   28.117
C   47.182   38.142   29.236
C   48.053   39.257   28.887
C   48.276   39.128   27.492
C   47.567   37.936   27.033
C   48.677   40.132   29.983
C   49.105   40.054   26.566
O   49.320   39.917   25.341
C   49.766   41.290   27.244
N   46.297   35.230   25.992
C   47.035   36.184   25.287
C   47.285   35.667   23.792
C   46.270   34.448   23.726
C   46.043   34.156   25.201
C   47.007   36.759   22.670
C   46.780   33.247   22.927
C   45.762   32.590   22.004
N   45.312   33.520   28.087
C   45.148   32.660   27.012
C   44.531   31.417   27.520
C   44.232   31.731   28.869
C   44.767   32.950   29.104
C   44.250   30.220   26.805
C   43.521   31.235   30.017
O   42.845   30.259   30.088
C   43.756   32.234   31.207
C   44.234   31.557   32.406
O   45.110   30.722   32.399
O   43.632   32.080   33.546
C   43.893   31.390   34.761
C   46.128   36.849   36.861
C   47.180   37.788   37.304
C   47.121   38.462   38.464
C   45.809   38.577   39.280
C   48.268   39.490   38.756
C   48.448   40.777   37.981
C   49.381   41.806   38.695
C   50.164   42.734   37.654
C   51.697   42.508   37.803
C   49.916   44.234   37.989
C   48.577   44.908   37.559
C   47.463   45.061   38.674
C   47.283   46.677   38.951
C   48.494   47.290   39.601
C   46.062   47.021   39.922
C   44.710   47.433   39.230
C   43.591   46.367   39.382
C   42.212   47.126   39.239
C   41.499   47.400   40.533
C   41.255   46.464   38.197
H   46.752   38.463   31.312
H   48.329   37.800   25.092
H   45.140   32.182   24.993
H   44.579   34.660   33.197
H   46.309   36.590   33.225
H   44.185   37.658   33.636
H   43.344   36.519   32.583
H   44.109   37.915   31.892
H   47.548   34.758   32.670
H   46.812   33.174   32.697
H   47.988   34.194   34.651
H   46.645   32.988   34.852
H   48.449   39.987   31.039
H   48.221   41.073   29.674
H   49.756   40.184   29.835
H   49.029   41.813   27.853
H   50.131   41.955   26.462
H   50.568   40.892   27.866
H   48.303   35.288   23.702
H   45.339   34.823   23.301
H   46.083   36.585   22.117
H   47.822   36.646   21.954
H   46.841   37.784   23.001
H   47.334   32.543   23.547
H   47.462   33.568   22.139
H   45.748   33.122   21.053
H   44.764   32.842   22.363
H   45.977   31.523   21.937
H   44.385   29.501   27.613
H   45.065   30.168   26.084
H   43.232   30.208   26.416
H   42.816   32.730   31.447
H   44.530   30.508   34.691
H   42.986   31.061   35.267
H   44.284   32.121   35.469
H   45.795   36.336   37.763
H   45.317   37.371   36.353
H   48.138   37.863   36.789
H   45.860   39.539   39.791
H   45.724   37.851   40.088
H   44.923   38.580   38.645
H   49.112   38.837   38.531
H   48.296   39.595   39.841
H   47.466   41.249   38.017
H   48.763   40.514   36.971
H   50.081   41.156   39.220
H   48.643   42.311   39.318
H   49.769   42.617   36.645
H   52.322   42.731   36.938
H   51.999   41.567   38.264
H   51.922   43.237   38.582
H   50.688   44.837   37.510
H   50.056   44.465   39.045
H   48.254   44.333   36.691
H   48.861   45.887   37.173
H   47.699   44.394   39.504
H   46.593   44.554   38.258
H   47.078   47.242   38.041
H   48.397   47.494   40.667
H   48.677   48.149   38.955
H   49.360   46.629   39.574
H   46.222   47.848   40.614
H   45.858   46.167   40.567
H   44.867   47.355   38.154
H   44.448   48.480   39.378
H   43.658   45.810   40.317
H   43.810   45.657   38.585
H   42.359   48.139   38.864
H   42.066   46.862   41.293
H   40.600   46.807   40.700
H   41.278   48.451   40.721
H   40.309   46.110   38.606
H   41.690   45.603   37.690
H   41.042   47.172   37.397

