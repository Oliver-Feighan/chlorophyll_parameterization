%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_801_chromophore_3 TDDFT with cam-b3lyp functional

0 1
Mg   1.912   8.086   26.362
C   2.180   10.164   29.288
C   2.378   5.363   28.638
C   2.369   5.981   23.745
C   2.103   10.730   24.297
N   2.176   7.851   28.789
C   2.128   8.811   29.696
C   2.110   8.216   31.113
C   2.462   6.661   30.846
C   2.364   6.580   29.338
C   3.803   6.048   31.538
C   0.859   8.416   31.860
C   0.675   7.534   33.127
C   1.875   7.304   34.087
O   2.853   8.056   34.161
O   1.565   6.177   34.766
N   2.177   5.955   26.244
C   2.453   5.067   27.237
C   2.778   3.710   26.725
C   2.609   3.803   25.314
C   2.340   5.247   25.029
C   3.078   2.567   27.694
C   2.715   2.710   24.273
O   2.436   2.948   23.101
C   3.193   1.286   24.649
N   2.003   8.276   24.356
C   2.090   7.322   23.385
C   1.826   7.877   21.910
C   1.594   9.398   22.202
C   1.892   9.447   23.682
C   3.009   7.520   21.001
C   0.139   9.921   21.880
C   0.173   10.878   20.675
N   2.065   10.012   26.662
C   2.140   11.021   25.716
C   2.207   12.366   26.411
C   2.281   12.095   27.755
C   2.084   10.625   27.877
C   2.086   13.684   25.834
C   2.355   12.674   29.068
O   2.513   13.876   29.411
C   2.244   11.436   30.104
C   3.402   11.459   31.095
O   4.607   11.232   30.854
O   2.993   11.970   32.246
C   3.917   11.925   33.398
C   2.575   5.809   35.745
C   2.070   4.788   36.778
C   2.153   4.639   38.123
C   3.039   5.574   39.026
C   1.230   3.731   38.875
C   1.071   2.255   38.454
C   1.366   1.378   39.655
C   2.621   0.525   39.519
C   3.657   1.199   40.391
C   2.370   -0.902   40.123
C   3.399   -1.976   39.746
C   4.205   -2.452   41.013
C   5.594   -3.121   40.626
C   5.543   -4.703   40.744
C   6.668   -2.638   41.534
C   7.498   -1.501   40.947
C   8.164   -0.687   42.033
C   9.476   -0.039   41.464
C   10.530   -0.121   42.566
C   9.216   1.359   40.953
H   2.623   4.656   29.434
H   2.499   5.327   22.879
H   2.154   11.603   23.643
H   2.858   8.634   31.787
H   1.575   6.100   31.140
H   4.069   6.710   32.362
H   4.591   5.874   30.805
H   3.571   5.071   31.961
H   0.015   8.108   31.243
H   0.722   9.453   32.168
H   0.249   6.553   32.916
H   -0.018   8.126   33.725
H   2.114   2.076   27.831
H   3.546   2.901   28.620
H   3.695   1.834   27.174
H   2.777   0.940   25.595
H   4.269   1.459   24.666
H   3.005   0.565   23.853
H   0.883   7.408   21.629
H   2.408   9.914   21.692
H   3.772   7.071   21.636
H   3.418   8.383   20.476
H   2.801   6.693   20.321
H   -0.231   10.455   22.755
H   -0.461   9.022   21.742
H   1.191   11.057   20.328
H   -0.239   11.844   20.965
H   -0.332   10.442   19.812
H   1.494   13.411   24.960
H   3.038   14.147   25.575
H   1.375   14.317   26.364
H   1.323   11.406   30.686
H   4.510   11.011   33.430
H   3.305   11.822   34.293
H   4.543   12.810   33.506
H   2.830   6.693   36.331
H   3.452   5.353   35.286
H   1.342   4.175   36.248
H   3.596   4.951   39.726
H   2.417   6.278   39.578
H   3.678   6.087   38.308
H   0.260   4.228   38.904
H   1.521   3.856   39.917
H   1.674   1.965   37.593
H   0.029   2.048   38.210
H   0.501   0.782   39.947
H   1.438   2.031   40.525
H   2.961   0.434   38.488
H   4.675   0.917   40.124
H   3.522   0.956   41.445
H   3.643   2.289   40.375
H   1.453   -1.227   39.631
H   2.214   -0.737   41.189
H   4.096   -1.685   38.960
H   2.809   -2.817   39.382
H   3.461   -3.131   41.429
H   4.330   -1.628   41.716
H   5.858   -2.997   39.575
H   5.934   -4.979   41.724
H   6.058   -5.247   39.953
H   4.475   -4.919   40.731
H   7.290   -3.532   41.571
H   6.384   -2.469   42.573
H   6.917   -0.848   40.296
H   8.281   -1.965   40.347
H   8.452   -1.301   42.886
H   7.338   -0.009   42.252
H   9.904   -0.518   40.584
H   10.018   0.251   43.453
H   11.394   0.515   42.368
H   10.788   -1.174   42.674
H   9.164   2.112   41.740
H   8.257   1.384   40.436
H   9.975   1.712   40.254

