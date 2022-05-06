%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_201_chromophore_11 TDDFT with PBE1PBE functional

0 1
Mg   52.800   24.054   43.828
C   49.855   25.808   43.642
C   51.081   21.074   43.163
C   55.756   22.569   43.276
C   54.433   27.157   44.420
N   50.781   23.587   43.272
C   49.780   24.455   43.357
C   48.454   23.702   43.159
C   48.804   22.191   43.359
C   50.324   22.216   43.240
C   48.261   21.547   44.648
C   47.783   23.948   41.703
C   48.663   23.784   40.500
C   47.933   23.414   39.284
O   47.099   24.142   38.714
O   48.216   22.118   38.969
N   53.379   22.094   43.302
C   52.515   20.957   43.247
C   53.339   19.736   43.224
C   54.714   20.179   43.176
C   54.639   21.670   43.240
C   52.789   18.375   43.114
C   55.877   19.131   43.147
O   55.653   17.924   43.080
C   57.276   19.612   43.220
N   54.804   24.840   43.734
C   55.832   23.987   43.566
C   57.156   24.734   43.705
C   56.754   26.077   44.330
C   55.238   26.054   44.136
C   58.275   23.881   44.521
C   57.459   27.257   43.562
C   58.037   28.304   44.512
N   52.289   26.141   44.071
C   53.058   27.258   44.349
C   52.172   28.324   44.633
C   50.882   27.772   44.310
C   51.039   26.524   43.927
C   52.531   29.735   45.143
C   49.446   28.084   44.267
O   48.766   29.114   44.505
C   48.771   26.871   43.732
C   48.127   27.211   42.439
O   48.797   27.246   41.415
O   46.795   27.421   42.559
C   46.043   27.987   41.403
C   47.387   21.491   38.005
C   48.380   21.397   36.803
C   48.392   20.440   35.891
C   47.172   19.590   35.688
C   49.565   20.363   34.901
C   51.031   20.434   35.336
C   51.928   19.630   34.327
C   52.595   18.410   34.976
C   53.954   18.658   35.654
C   52.858   17.448   33.711
C   51.636   16.560   33.355
C   51.711   15.010   33.515
C   51.679   14.194   32.228
C   51.538   12.676   32.473
C   53.042   14.467   31.398
C   52.971   14.433   29.872
C   54.039   13.424   29.304
C   54.048   13.649   27.765
C   54.005   12.348   27.048
C   55.412   14.431   27.377
H   50.595   20.101   43.062
H   56.744   22.105   43.237
H   55.038   28.000   44.762
H   47.797   23.995   43.978
H   48.443   21.616   42.506
H   48.048   20.501   44.425
H   47.277   21.917   44.935
H   48.953   21.520   45.490
H   47.292   24.918   41.635
H   46.924   23.287   41.585
H   49.342   22.962   40.727
H   49.217   24.716   40.389
H   53.263   17.881   43.963
H   53.062   17.816   42.219
H   51.725   18.366   43.351
H   57.523   20.181   42.324
H   57.871   18.699   43.201
H   57.509   20.245   44.076
H   57.609   24.929   42.733
H   57.095   26.106   45.365
H   59.093   23.455   43.939
H   57.786   23.011   44.957
H   58.710   24.487   45.316
H   56.760   27.829   42.950
H   58.385   26.884   43.126
H   58.698   27.799   45.217
H   57.244   28.859   45.012
H   58.642   28.946   43.871
H   52.298   30.540   44.445
H   53.621   29.731   45.177
H   52.066   29.886   46.118
H   48.028   26.437   44.401
H   46.725   28.115   40.563
H   45.601   28.949   41.660
H   45.266   27.388   40.928
H   46.537   22.115   37.728
H   47.083   20.498   38.336
H   49.287   21.982   36.956
H   47.082   18.879   36.509
H   47.195   18.929   34.821
H   46.337   20.288   35.635
H   49.401   21.257   34.298
H   49.346   19.478   34.304
H   51.189   20.187   36.386
H   51.288   21.475   35.138
H   52.743   20.263   33.977
H   51.350   19.452   33.420
H   51.854   17.964   35.640
H   53.975   19.740   35.791
H   54.829   18.352   35.082
H   53.931   18.089   36.583
H   53.705   16.800   33.937
H   53.051   18.075   32.841
H   51.268   16.645   32.332
H   50.809   16.991   33.919
H   51.121   14.689   34.374
H   52.708   14.716   33.842
H   50.990   14.554   31.464
H   52.210   12.050   31.886
H   50.534   12.287   32.302
H   51.720   12.461   33.526
H   53.801   13.783   31.777
H   53.377   15.494   31.546
H   53.139   15.450   29.518
H   51.982   14.118   29.538
H   53.802   12.411   29.629
H   55.046   13.568   29.695
H   53.287   14.362   27.447
H   54.001   11.503   27.736
H   54.928   12.177   26.493
H   53.212   12.430   26.305
H   55.352   15.484   27.650
H   55.686   14.446   26.322
H   56.247   14.000   27.929

