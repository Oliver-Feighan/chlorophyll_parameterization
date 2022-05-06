%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1801_chromophore_14 TDDFT with PBE1PBE functional

0 1
Mg   46.824   45.880   43.390
C   43.488   44.795   42.835
C   47.906   42.879   42.328
C   49.993   47.289   42.781
C   45.508   49.270   43.481
N   45.833   44.115   42.384
C   44.474   43.821   42.543
C   44.133   42.433   42.010
C   45.510   41.739   42.183
C   46.489   42.983   42.300
C   45.605   40.648   43.282
C   43.610   42.520   40.540
C   42.303   41.790   40.160
C   42.365   41.049   38.832
O   42.488   39.840   38.627
O   42.179   41.986   37.812
N   48.699   45.066   42.861
C   48.893   43.777   42.538
C   50.312   43.557   42.600
C   50.929   44.813   42.767
C   49.854   45.816   42.818
C   50.952   42.208   42.568
C   52.494   45.081   42.914
O   53.296   44.164   42.735
C   53.031   46.432   43.237
N   47.568   47.917   43.081
C   48.926   48.224   42.739
C   49.173   49.706   42.601
C   47.803   50.348   42.864
C   46.885   49.113   43.176
C   50.444   50.240   43.437
C   47.197   51.277   41.767
C   47.323   52.836   42.083
N   44.957   46.853   43.474
C   44.598   48.179   43.528
C   43.112   48.234   43.708
C   42.655   46.930   43.443
C   43.798   46.113   43.234
C   42.327   49.458   44.251
C   41.451   46.109   43.334
O   40.276   46.416   43.343
C   41.952   44.699   42.809
C   41.407   43.711   43.814
O   40.620   42.801   43.527
O   41.924   43.863   45.031
C   41.461   42.965   46.093
C   42.257   41.494   36.385
C   43.222   42.448   35.687
C   44.443   42.194   35.177
C   45.075   40.777   35.111
C   45.111   43.286   34.331
C   44.955   43.268   32.827
C   45.112   44.717   32.211
C   46.090   44.659   31.013
C   47.006   45.897   30.986
C   45.440   44.513   29.604
C   45.099   43.023   29.270
C   43.882   43.143   28.284
C   44.299   43.008   26.802
C   43.411   43.951   25.845
C   44.249   41.564   26.274
C   45.411   41.230   25.337
C   44.919   41.099   23.870
C   46.002   41.534   22.832
C   45.994   42.993   22.646
C   45.703   40.824   21.490
H   48.115   41.834   42.092
H   50.935   47.801   42.574
H   45.057   50.262   43.414
H   43.354   41.908   42.563
H   45.851   41.251   41.270
H   46.517   40.737   43.872
H   45.562   39.651   42.843
H   44.771   40.755   43.977
H   44.459   42.393   39.867
H   43.408   43.589   40.466
H   41.492   42.518   40.141
H   42.090   41.101   40.977
H   51.849   42.207   43.188
H   51.370   41.977   41.589
H   50.213   41.440   42.798
H   52.643   46.700   44.219
H   52.974   47.169   42.436
H   54.101   46.404   43.443
H   49.396   49.732   41.534
H   47.850   51.021   43.720
H   50.682   49.507   44.207
H   50.352   51.282   43.745
H   51.285   50.193   42.745
H   46.126   51.097   41.677
H   47.671   51.068   40.807
H   46.590   53.247   42.778
H   47.060   53.353   41.160
H   48.353   53.015   42.394
H   42.651   49.731   45.255
H   41.265   49.223   44.327
H   42.397   50.384   43.681
H   41.644   44.415   41.803
H   40.600   43.514   46.474
H   42.235   42.855   46.853
H   41.103   41.957   45.884
H   41.221   41.667   36.094
H   42.402   40.438   36.158
H   42.849   43.432   35.401
H   45.250   40.359   34.119
H   44.458   40.083   35.681
H   46.015   40.788   35.662
H   46.166   43.037   34.437
H   44.929   44.266   34.773
H   43.910   42.985   32.703
H   45.506   42.542   32.230
H   45.558   45.387   32.947
H   44.111   45.104   32.020
H   46.735   43.805   31.223
H   47.820   45.863   31.710
H   46.492   46.857   30.952
H   47.551   45.848   30.043
H   46.084   44.838   28.786
H   44.589   45.194   29.603
H   44.834   42.509   30.195
H   46.013   42.609   28.845
H   43.427   44.125   28.413
H   43.159   42.389   28.595
H   45.309   43.405   26.702
H   43.957   44.846   25.547
H   42.453   44.226   26.287
H   43.235   43.478   24.879
H   43.241   41.432   25.880
H   44.253   40.850   27.097
H   45.734   40.214   25.566
H   46.339   41.800   25.373
H   44.036   41.730   23.774
H   44.549   40.091   23.682
H   46.976   41.287   23.254
H   46.965   43.420   22.397
H   45.666   43.570   23.511
H   45.295   43.216   21.839
H   44.680   40.447   21.477
H   46.448   40.034   21.587
H   45.835   41.438   20.599
