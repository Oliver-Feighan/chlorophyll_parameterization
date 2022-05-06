%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_5 TDDFT with cam-b3lyp functional

0 1
Mg   24.322   -6.148   46.115
C   26.649   -3.789   44.917
C   21.949   -4.563   44.251
C   22.371   -8.878   46.399
C   27.055   -7.739   47.639
N   24.381   -4.362   44.693
C   25.389   -3.579   44.379
C   24.899   -2.494   43.337
C   23.329   -2.487   43.561
C   23.171   -3.889   44.235
C   22.564   -1.405   44.285
C   25.440   -2.667   41.810
C   25.881   -1.446   41.154
C   24.988   -0.851   39.998
O   23.961   -0.198   40.174
O   25.379   -1.447   38.858
N   22.410   -6.562   45.592
C   21.591   -5.740   44.841
C   20.230   -6.249   44.781
C   20.363   -7.534   45.301
C   21.741   -7.712   45.827
C   19.023   -5.568   44.053
C   19.198   -8.552   45.344
O   18.097   -8.199   44.978
C   19.438   -10.030   45.753
N   24.688   -8.067   46.867
C   23.712   -9.070   46.940
C   24.334   -10.388   47.532
C   25.749   -9.922   47.953
C   25.861   -8.437   47.465
C   23.498   -11.018   48.719
C   26.867   -10.873   47.425
C   27.624   -11.622   48.544
N   26.460   -5.935   46.256
C   27.413   -6.560   47.041
C   28.715   -5.967   46.909
C   28.447   -4.840   46.120
C   27.049   -4.881   45.769
C   29.950   -6.588   47.417
C   28.988   -3.695   45.508
O   30.108   -3.209   45.603
C   27.920   -3.060   44.628
C   27.849   -1.577   44.920
O   27.978   -0.682   44.121
O   27.391   -1.352   46.254
C   27.023   0.056   46.523
C   24.324   -1.475   37.767
C   24.144   -2.925   37.450
C   24.636   -3.684   36.466
C   25.650   -3.351   35.405
C   24.227   -5.167   36.400
C   22.903   -5.563   35.596
C   22.996   -6.791   34.675
C   22.592   -6.367   33.273
C   22.192   -7.616   32.438
C   23.735   -5.575   32.577
C   23.327   -4.162   32.050
C   24.473   -3.254   31.489
C   24.193   -2.922   29.988
C   23.106   -1.901   29.760
C   25.503   -2.645   29.226
C   25.922   -3.777   28.260
C   25.689   -3.499   26.695
C   24.530   -4.369   26.085
C   24.779   -4.722   24.557
C   23.188   -3.703   26.417
H   21.192   -4.013   43.687
H   21.803   -9.793   46.577
H   27.779   -8.359   48.173
H   25.390   -1.606   43.734
H   22.864   -2.520   42.576
H   21.638   -1.108   43.793
H   23.138   -0.483   44.188
H   22.406   -1.598   45.346
H   24.726   -3.204   41.185
H   26.299   -3.328   41.932
H   26.917   -1.496   40.816
H   25.964   -0.673   41.918
H   18.221   -6.221   43.708
H   19.378   -5.115   43.127
H   18.666   -4.796   44.734
H   19.584   -10.319   46.794
H   20.273   -10.395   45.154
H   18.614   -10.616   45.347
H   24.326   -11.103   46.709
H   25.932   -9.829   49.023
H   22.482   -10.661   48.883
H   24.013   -10.917   49.674
H   23.236   -12.040   48.443
H   27.640   -10.356   46.857
H   26.502   -11.646   46.748
H   28.595   -11.131   48.619
H   27.744   -12.655   48.220
H   26.940   -11.639   49.392
H   29.923   -6.719   48.499
H   30.816   -5.935   47.309
H   30.127   -7.622   47.122
H   28.348   -3.094   43.626
H   26.511   -0.112   47.470
H   26.277   0.441   45.827
H   27.970   0.594   46.557
H   24.610   -0.985   36.836
H   23.339   -1.140   38.091
H   23.376   -3.407   38.055
H   26.555   -3.925   35.604
H   25.901   -2.290   35.407
H   25.111   -3.590   34.488
H   24.164   -5.521   37.429
H   25.066   -5.654   35.902
H   22.664   -4.642   35.064
H   22.111   -5.694   36.333
H   22.386   -7.609   35.059
H   23.987   -7.213   34.508
H   21.773   -5.650   33.307
H   21.178   -7.560   32.042
H   22.108   -8.494   33.080
H   22.933   -7.853   31.676
H   24.059   -6.115   31.687
H   24.672   -5.354   33.089
H   22.768   -3.512   32.723
H   22.759   -4.369   31.143
H   25.460   -3.716   31.521
H   24.407   -2.368   32.120
H   23.754   -3.800   29.513
H   22.252   -2.397   29.297
H   23.372   -1.087   29.086
H   22.818   -1.338   30.648
H   26.313   -2.510   29.944
H   25.371   -1.652   28.796
H   25.353   -4.654   28.572
H   27.010   -3.792   28.315
H   26.657   -3.666   26.223
H   25.434   -2.452   26.530
H   24.394   -5.384   26.459
H   25.599   -4.138   24.138
H   23.902   -4.613   23.918
H   25.028   -5.776   24.439
H   22.634   -3.351   25.547
H   23.267   -2.774   26.982
H   22.508   -4.386   26.928

