%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_801_chromophore_7 TDDFT with PBE1PBE functional

0 1
Mg   26.357   -0.043   29.416
C   28.165   -0.315   32.419
C   23.482   0.236   31.275
C   24.718   -0.205   26.624
C   29.352   -0.454   27.678
N   25.889   0.084   31.592
C   26.813   0.002   32.641
C   26.152   0.014   34.051
C   24.582   0.080   33.594
C   24.644   0.058   32.076
C   23.747   -1.139   34.159
C   26.624   1.302   34.833
C   25.841   1.892   36.006
C   26.622   2.391   37.208
O   27.677   3.015   37.186
O   26.072   1.933   38.388
N   24.344   0.032   28.922
C   23.355   0.358   29.837
C   22.131   0.690   29.105
C   22.400   0.449   27.694
C   23.839   -0.041   27.718
C   20.828   1.016   29.816
C   21.544   0.791   26.457
O   22.040   0.805   25.316
C   20.069   1.150   26.608
N   26.980   -0.402   27.428
C   26.105   -0.379   26.456
C   26.842   -0.413   25.084
C   28.332   -0.083   25.458
C   28.197   -0.318   26.955
C   26.636   -1.796   24.317
C   28.655   1.406   25.148
C   29.655   1.631   23.967
N   28.333   -0.297   29.867
C   29.436   -0.529   29.093
C   30.554   -0.878   29.919
C   30.120   -0.835   31.245
C   28.774   -0.420   31.164
C   31.904   -1.289   29.532
C   30.538   -0.890   32.670
O   31.669   -1.044   33.172
C   29.274   -0.440   33.451
C   29.056   -1.446   34.517
O   28.579   -2.562   34.440
O   29.417   -0.884   35.699
C   29.395   -1.755   36.868
C   26.580   2.648   39.548
C   25.596   2.966   40.633
C   25.696   2.933   41.927
C   27.005   2.684   42.734
C   24.497   3.244   42.783
C   24.272   2.386   44.092
C   22.797   2.397   44.520
C   22.659   2.455   46.061
C   21.353   1.850   46.553
C   22.696   4.002   46.519
C   23.851   4.414   47.484
C   23.380   4.695   48.980
C   23.434   6.181   49.253
C   24.296   6.611   50.387
C   22.013   6.859   49.336
C   22.182   8.433   49.124
C   21.432   9.121   47.962
C   22.317   9.304   46.697
C   23.136   10.590   46.792
C   21.409   9.392   45.441
H   22.591   0.402   31.883
H   24.226   -0.090   25.656
H   30.206   -0.491   26.999
H   26.504   -0.891   34.547
H   24.142   1.001   33.977
H   22.848   -0.806   34.678
H   24.363   -1.707   34.856
H   23.504   -1.796   33.324
H   26.895   2.081   34.121
H   27.595   1.050   35.259
H   25.059   1.232   36.379
H   25.293   2.784   35.703
H   20.148   0.255   29.433
H   20.527   2.021   29.523
H   20.790   0.950   30.903
H   19.554   1.249   25.653
H   19.981   2.110   27.118
H   19.498   0.450   27.218
H   26.329   0.306   24.445
H   29.003   -0.835   25.045
H   27.570   -2.267   24.009
H   25.956   -1.570   23.497
H   26.224   -2.560   24.977
H   29.224   1.835   25.973
H   27.763   2.013   24.991
H   30.617   1.807   24.449
H   29.312   2.464   23.353
H   29.735   0.793   23.275
H   32.096   -1.341   28.460
H   32.014   -2.273   29.987
H   32.582   -0.610   30.048
H   29.408   0.527   33.937
H   29.376   -2.842   36.789
H   28.479   -1.500   37.401
H   30.226   -1.387   37.470
H   27.226   3.512   39.391
H   27.255   1.833   39.806
H   24.603   3.272   40.303
H   26.994   1.717   43.237
H   27.257   3.481   43.433
H   27.789   2.653   41.977
H   23.601   3.197   42.164
H   24.595   4.290   43.072
H   24.806   2.911   44.884
H   24.540   1.339   43.956
H   22.228   1.541   44.157
H   22.381   3.311   44.096
H   23.455   1.890   46.545
H   20.592   1.465   45.873
H   20.788   2.572   47.142
H   21.578   0.973   47.160
H   21.694   4.279   46.848
H   22.825   4.717   45.706
H   24.460   5.187   47.016
H   24.488   3.546   47.654
H   23.905   4.101   49.728
H   22.328   4.425   49.070
H   23.845   6.642   48.355
H   23.686   6.884   51.248
H   24.829   7.467   49.973
H   24.985   5.910   50.857
H   21.666   6.761   50.365
H   21.267   6.483   48.637
H   23.237   8.708   49.114
H   21.951   8.837   50.110
H   21.107   10.083   48.359
H   20.588   8.484   47.695
H   22.937   8.410   46.635
H   22.572   11.241   47.460
H   23.301   11.002   45.797
H   24.036   10.323   47.347
H   21.785   8.687   44.700
H   21.412   10.328   44.882
H   20.404   9.010   45.618

