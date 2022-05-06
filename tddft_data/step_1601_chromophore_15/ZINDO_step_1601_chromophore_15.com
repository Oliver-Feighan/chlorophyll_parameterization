%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1601_chromophore_15 ZINDO

0 1
Mg   47.004   34.643   28.354
C   45.356   32.622   30.573
C   46.759   37.241   30.569
C   48.081   36.639   25.939
C   46.562   32.172   25.941
N   46.153   34.869   30.307
C   45.546   33.948   31.052
C   45.297   34.453   32.551
C   45.409   36.024   32.312
C   46.187   36.074   30.985
C   44.045   36.785   32.220
C   46.327   33.767   33.575
C   47.463   34.577   34.358
C   47.196   34.818   35.860
O   47.261   33.891   36.635
O   46.790   36.131   36.113
N   47.556   36.624   28.292
C   47.450   37.531   29.335
C   48.102   38.776   28.938
C   48.431   38.626   27.581
C   48.012   37.266   27.221
C   48.305   39.946   29.880
C   49.069   39.611   26.630
O   49.406   39.423   25.481
C   49.344   41.014   27.180
N   47.183   34.470   26.210
C   47.642   35.410   25.454
C   47.902   35.019   23.982
C   47.383   33.552   23.936
C   46.872   33.380   25.442
C   47.177   35.905   22.961
C   48.352   32.428   23.379
C   47.776   31.513   22.245
N   46.266   32.719   28.270
C   46.273   31.775   27.309
C   45.770   30.508   27.834
C   45.376   30.787   29.106
C   45.670   32.162   29.284
C   45.820   29.238   27.074
C   44.655   30.260   30.291
O   44.146   29.207   30.536
C   44.603   31.450   31.251
C   45.210   31.030   32.539
O   46.334   30.461   32.530
O   44.431   31.215   33.616
C   45.143   30.758   34.822
C   46.447   36.440   37.539
C   47.390   37.495   37.988
C   47.156   38.556   38.786
C   45.916   38.756   39.547
C   48.262   39.602   38.923
C   48.065   40.789   37.922
C   49.311   41.058   36.996
C   49.307   42.608   36.661
C   49.582   42.739   35.077
C   50.304   43.461   37.505
C   49.590   44.330   38.596
C   49.481   45.826   38.191
C   47.997   46.348   38.118
C   47.972   47.868   37.679
C   47.106   46.014   39.320
C   45.876   45.161   39.046
C   44.802   45.951   38.206
C   43.650   46.353   39.198
C   42.319   45.839   38.719
C   43.610   47.831   39.299
H   46.764   38.110   31.230
H   48.403   37.231   25.079
H   46.517   31.390   25.181
H   44.294   34.178   32.877
H   45.972   36.587   33.056
H   44.193   37.772   31.783
H   43.612   36.902   33.213
H   43.456   36.186   31.525
H   46.884   32.941   33.133
H   45.636   33.536   34.387
H   47.854   35.486   33.900
H   48.336   33.925   34.324
H   49.389   39.959   29.990
H   47.834   39.808   30.853
H   48.009   40.949   29.572
H   50.136   41.015   27.929
H   48.417   41.434   27.569
H   49.598   41.687   26.361
H   48.977   35.082   23.814
H   46.441   33.560   23.387
H   46.495   36.480   23.588
H   46.587   35.408   22.190
H   47.757   36.642   22.406
H   48.668   31.812   24.221
H   49.190   32.940   22.905
H   46.859   31.926   21.825
H   47.698   30.484   22.598
H   48.417   31.441   21.367
H   46.736   29.279   26.485
H   44.960   29.170   26.408
H   45.796   28.312   27.647
H   43.623   31.799   31.577
H   44.772   31.401   35.620
H   46.210   30.977   34.859
H   44.972   29.697   35.005
H   46.450   35.602   38.236
H   45.465   36.913   37.527
H   48.333   37.493   37.442
H   45.928   38.823   40.635
H   45.173   37.974   39.390
H   45.561   39.729   39.206
H   49.262   39.214   38.732
H   48.216   40.004   39.935
H   47.886   41.722   38.456
H   47.214   40.548   37.285
H   49.296   40.368   36.152
H   50.240   40.824   37.516
H   48.271   42.923   36.784
H   50.604   42.705   34.698
H   49.147   43.699   34.801
H   49.021   41.967   34.550
H   50.740   44.163   36.794
H   51.095   42.775   37.808
H   50.174   44.314   39.516
H   48.584   43.945   38.764
H   50.129   46.097   37.356
H   49.812   46.372   39.074
H   47.635   45.789   37.255
H   48.866   48.320   38.108
H   47.066   48.432   37.900
H   48.002   47.862   36.590
H   46.847   46.959   39.797
H   47.620   45.467   40.110
H   45.412   44.829   39.975
H   46.216   44.252   38.550
H   44.488   45.341   37.359
H   45.276   46.807   37.726
H   43.706   45.960   40.213
H   42.058   46.292   37.762
H   41.551   45.864   39.492
H   42.524   44.794   38.488
H   44.586   48.175   39.642
H   42.874   48.145   40.039
H   43.387   48.216   38.304
