%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1601_chromophore_25 TDDFT with blyp functional

0 1
Mg   -3.276   34.102   26.550
C   -4.336   32.298   29.380
C   -1.976   36.404   28.677
C   -3.007   36.291   23.896
C   -5.072   31.940   24.444
N   -3.202   34.253   28.775
C   -3.463   33.326   29.756
C   -3.010   33.764   31.133
C   -2.269   35.145   30.842
C   -2.488   35.295   29.329
C   -2.506   36.245   31.864
C   -2.198   32.646   31.861
C   -2.729   32.136   33.222
C   -1.800   32.235   34.404
O   -0.794   31.589   34.507
O   -2.249   33.229   35.273
N   -2.409   35.948   26.347
C   -1.810   36.707   27.325
C   -1.111   37.852   26.725
C   -1.433   37.782   25.326
C   -2.357   36.652   25.116
C   -0.267   38.852   27.549
C   -0.862   38.608   24.209
O   -1.052   38.237   23.050
C   -0.091   39.867   24.489
N   -4.002   34.181   24.435
C   -3.772   35.198   23.628
C   -4.110   34.811   22.164
C   -4.637   33.331   22.327
C   -4.526   33.080   23.804
C   -5.215   35.759   21.745
C   -3.745   32.335   21.534
C   -2.357   32.015   22.096
N   -4.405   32.441   26.797
C   -5.063   31.669   25.841
C   -5.923   30.735   26.541
C   -5.587   30.933   27.898
C   -4.686   31.967   27.999
C   -6.970   29.760   25.978
C   -5.955   30.455   29.233
O   -6.676   29.568   29.620
C   -4.983   31.310   30.226
C   -5.939   31.949   31.103
O   -6.535   33.021   30.890
O   -5.992   31.291   32.276
C   -6.520   32.029   33.438
C   -1.307   33.572   36.302
C   -2.014   34.376   37.385
C   -1.518   34.411   38.643
C   -0.194   33.793   39.136
C   -2.313   35.063   39.772
C   -1.824   36.439   40.232
C   -0.849   36.324   41.542
C   -1.410   37.194   42.745
C   -0.875   38.664   42.615
C   -1.203   36.509   44.115
C   0.284   35.975   44.232
C   0.962   36.484   45.575
C   1.963   35.444   46.285
C   1.475   35.002   47.644
C   3.440   35.894   46.251
C   4.265   34.911   45.346
C   5.082   35.521   44.177
C   4.528   34.927   42.874
C   5.197   33.562   42.478
C   4.522   35.936   41.750
H   -1.467   37.075   29.372
H   -2.889   36.956   23.038
H   -5.773   31.403   23.801
H   -3.909   33.963   31.717
H   -1.218   34.870   30.940
H   -1.533   36.477   32.299
H   -3.240   35.984   32.626
H   -2.889   37.202   31.507
H   -1.169   32.989   31.969
H   -2.130   31.707   31.311
H   -2.813   31.087   32.937
H   -3.652   32.638   33.512
H   -0.302   38.529   28.589
H   -0.810   39.796   27.523
H   0.770   38.879   27.212
H   -0.692   40.498   25.143
H   0.101   40.340   23.526
H   0.885   39.614   24.905
H   -3.142   34.919   21.673
H   -5.674   33.206   22.015
H   -4.897   36.282   20.843
H   -5.464   36.428   22.568
H   -6.094   35.125   21.623
H   -3.631   32.842   20.576
H   -4.241   31.404   21.259
H   -2.035   32.674   22.903
H   -1.659   32.051   21.260
H   -2.327   30.967   22.395
H   -7.657   29.264   26.664
H   -6.383   29.086   25.354
H   -7.550   30.273   25.211
H   -4.259   30.788   30.852
H   -6.008   32.977   33.598
H   -6.454   31.393   34.321
H   -7.580   32.166   33.223
H   -0.348   33.975   35.978
H   -0.881   32.736   36.856
H   -3.083   34.580   37.343
H   0.377   33.140   38.475
H   -0.485   33.053   39.882
H   0.409   34.589   39.571
H   -2.435   34.367   40.602
H   -3.300   35.267   39.356
H   -2.646   37.154   40.249
H   -1.137   36.760   39.449
H   0.197   36.558   41.342
H   -0.739   35.263   41.763
H   -2.482   37.374   42.671
H   -1.630   39.432   42.443
H   -0.058   38.735   41.896
H   -0.340   39.011   43.499
H   -1.878   35.671   44.291
H   -1.466   37.259   44.861
H   0.931   36.246   43.397
H   0.237   34.889   44.153
H   0.105   36.732   46.202
H   1.452   37.419   45.302
H   1.827   34.557   45.666
H   0.807   34.142   47.636
H   1.012   35.923   47.999
H   2.352   34.819   48.265
H   3.829   35.927   47.269
H   3.578   36.886   45.820
H   3.657   34.089   44.969
H   5.015   34.463   45.997
H   6.166   35.431   44.241
H   4.880   36.591   44.205
H   3.486   34.637   43.011
H   4.440   32.778   42.444
H   5.962   33.372   43.231
H   5.653   33.749   41.506
H   4.453   35.517   40.746
H   5.378   36.610   41.773
H   3.605   36.518   41.842

