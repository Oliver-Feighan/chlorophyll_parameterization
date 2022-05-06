%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1301_chromophore_21 TDDFT with cam-b3lyp functional

0 1
Mg   15.957   53.276   24.993
C   17.459   51.960   27.799
C   13.345   54.344   26.895
C   14.426   54.213   22.143
C   18.345   51.450   23.111
N   15.433   53.014   27.065
C   16.249   52.701   28.083
C   15.707   53.119   29.425
C   14.214   53.582   29.092
C   14.254   53.575   27.548
C   13.124   52.497   29.606
C   16.649   54.212   30.060
C   16.988   53.998   31.532
C   16.035   53.220   32.460
O   16.177   52.102   32.888
O   14.848   53.876   32.690
N   14.191   54.375   24.634
C   13.255   54.737   25.550
C   12.209   55.414   24.849
C   12.544   55.395   23.470
C   13.799   54.658   23.330
C   11.008   55.929   25.630
C   11.789   55.885   22.247
O   12.165   55.774   21.067
C   10.452   56.467   22.428
N   16.288   52.820   22.908
C   15.481   53.302   21.925
C   16.121   53.039   20.551
C   17.467   52.205   20.888
C   17.407   52.216   22.392
C   15.269   52.315   19.482
C   18.779   52.839   20.227
C   19.301   52.188   18.932
N   17.484   51.975   25.332
C   18.431   51.395   24.514
C   19.401   50.671   25.172
C   19.086   50.955   26.501
C   17.857   51.651   26.529
C   20.514   49.829   24.575
C   19.586   50.794   27.803
O   20.688   50.362   28.136
C   18.509   51.408   28.778
C   18.093   50.428   29.724
O   17.140   49.625   29.546
O   18.910   50.590   30.849
C   18.436   49.882   32.035
C   13.848   53.209   33.422
C   12.531   53.906   33.222
C   11.299   53.433   33.344
C   10.944   52.091   33.866
C   10.227   54.440   33.314
C   9.677   54.864   34.641
C   9.709   56.408   34.797
C   8.363   57.082   35.044
C   8.575   58.612   34.831
C   7.706   56.852   36.388
C   6.141   56.961   36.381
C   5.672   57.386   37.702
C   4.422   58.331   37.499
C   4.894   59.775   37.093
C   3.467   58.193   38.704
C   2.007   57.886   38.324
C   0.987   58.398   39.325
C   -0.019   59.418   38.595
C   -0.688   60.595   39.534
C   -1.186   58.698   37.858
H   12.630   54.781   27.595
H   14.004   54.507   21.180
H   19.078   50.927   22.494
H   15.590   52.217   30.026
H   13.936   54.555   29.497
H   12.625   52.682   30.557
H   13.532   51.492   29.717
H   12.422   52.566   28.775
H   16.243   55.223   30.056
H   17.615   54.239   29.554
H   17.210   54.992   31.921
H   17.990   53.571   31.566
H   10.277   55.131   25.763
H   10.537   56.739   25.073
H   11.275   56.379   26.585
H   10.120   56.664   21.409
H   10.554   57.427   22.935
H   9.662   55.899   22.920
H   16.455   54.016   20.201
H   17.262   51.176   20.595
H   15.751   51.718   18.708
H   14.964   53.185   18.899
H   14.428   51.811   19.959
H   19.545   52.899   20.999
H   18.543   53.853   19.905
H   18.576   51.555   18.420
H   20.125   51.555   19.263
H   19.636   52.948   18.226
H   20.328   49.315   23.632
H   20.466   49.132   25.411
H   21.410   50.442   24.480
H   18.984   52.196   29.363
H   18.689   48.833   31.880
H   17.368   49.876   32.252
H   18.912   50.299   32.922
H   14.138   53.386   34.458
H   13.768   52.158   33.145
H   12.549   54.894   32.762
H   10.949   51.411   33.014
H   9.944   51.978   34.286
H   11.681   51.723   34.579
H   9.457   53.903   32.759
H   10.568   55.217   32.630
H   10.301   54.428   35.422
H   8.706   54.399   34.809
H   10.114   56.954   33.945
H   10.374   56.719   35.603
H   7.702   56.736   34.249
H   7.632   59.154   34.903
H   8.952   58.807   33.827
H   9.341   59.068   35.457
H   8.140   57.474   37.170
H   7.818   55.830   36.751
H   5.698   55.978   36.226
H   5.903   57.722   35.638
H   6.410   58.017   38.199
H   5.348   56.517   38.274
H   3.918   57.801   36.690
H   5.969   59.880   37.245
H   4.293   60.483   37.663
H   4.593   59.927   36.057
H   3.573   59.074   39.338
H   3.643   57.406   39.437
H   1.872   56.805   38.317
H   1.795   58.269   37.326
H   1.587   58.943   40.054
H   0.507   57.542   39.800
H   0.606   60.031   37.945
H   -0.580   61.589   39.101
H   -0.280   60.520   40.542
H   -1.753   60.447   39.713
H   -1.502   59.328   37.027
H   -2.008   58.360   38.489
H   -0.808   57.857   37.278

