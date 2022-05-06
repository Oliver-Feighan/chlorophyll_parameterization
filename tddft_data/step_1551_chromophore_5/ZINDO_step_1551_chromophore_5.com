%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1551_chromophore_5 ZINDO

0 1
Mg   24.726   -6.656   46.930
C   27.168   -4.381   46.078
C   22.636   -5.065   44.690
C   22.547   -9.146   47.251
C   27.130   -8.367   48.745
N   24.973   -5.043   45.383
C   26.015   -4.227   45.269
C   25.848   -3.197   44.169
C   24.371   -3.363   43.727
C   23.911   -4.501   44.688
C   23.610   -1.976   44.027
C   26.730   -3.611   42.942
C   26.917   -2.417   41.937
C   25.940   -2.468   40.757
O   25.010   -1.668   40.555
O   26.099   -3.613   40.030
N   22.808   -7.091   46.055
C   22.070   -6.218   45.283
C   20.762   -6.732   45.074
C   20.679   -7.935   45.777
C   22.070   -8.112   46.410
C   19.746   -6.049   44.295
C   19.447   -8.907   45.928
O   18.359   -8.673   45.440
C   19.593   -10.132   46.755
N   24.844   -8.537   47.823
C   23.792   -9.377   47.926
C   24.077   -10.615   48.861
C   25.481   -10.174   49.413
C   25.870   -8.985   48.554
C   23.077   -10.924   49.987
C   26.541   -11.336   49.357
C   27.162   -11.698   50.714
N   26.722   -6.455   47.441
C   27.591   -7.237   48.150
C   28.877   -6.574   48.231
C   28.728   -5.454   47.466
C   27.434   -5.441   46.958
C   30.147   -7.063   48.928
C   29.426   -4.232   47.058
O   30.515   -3.783   47.397
C   28.461   -3.490   46.130
C   28.189   -2.224   46.777
O   28.663   -1.136   46.552
O   27.181   -2.490   47.708
C   26.414   -1.298   48.072
C   25.270   -3.708   38.835
C   26.177   -4.557   37.864
C   25.838   -5.055   36.637
C   24.488   -4.771   36.038
C   26.856   -5.844   35.806
C   26.787   -7.323   35.951
C   26.743   -7.943   34.579
C   25.347   -7.929   33.971
C   24.585   -9.305   34.134
C   25.297   -7.246   32.545
C   24.383   -6.049   32.369
C   24.988   -4.668   31.884
C   24.547   -4.465   30.434
C   23.173   -3.859   30.367
C   25.553   -3.613   29.654
C   26.351   -4.481   28.584
C   26.079   -4.122   27.110
C   25.213   -5.146   26.320
C   25.674   -5.267   24.822
C   23.718   -4.846   26.421
H   21.893   -4.604   44.036
H   21.758   -9.829   47.573
H   27.766   -8.917   49.442
H   26.214   -2.244   44.550
H   24.215   -3.576   42.670
H   24.226   -1.216   44.510
H   22.792   -2.070   44.741
H   23.317   -1.536   43.074
H   26.148   -4.361   42.407
H   27.667   -4.095   43.216
H   27.938   -2.344   41.562
H   26.889   -1.436   42.409
H   19.056   -6.783   43.879
H   20.184   -5.485   43.471
H   19.218   -5.332   44.925
H   19.840   -9.902   47.792
H   20.311   -10.752   46.219
H   18.616   -10.612   46.690
H   24.128   -11.530   48.272
H   25.294   -9.797   50.418
H   22.712   -11.943   49.857
H   22.228   -10.259   49.826
H   23.496   -10.697   50.967
H   27.394   -11.230   48.685
H   26.124   -12.244   48.920
H   28.177   -11.306   50.766
H   27.195   -12.777   50.867
H   26.449   -11.286   51.428
H   30.096   -8.147   48.823
H   30.038   -6.885   49.998
H   31.042   -6.586   48.529
H   28.995   -3.375   45.187
H   26.307   -0.659   47.196
H   26.940   -0.766   48.865
H   25.470   -1.736   48.397
H   25.113   -2.743   38.352
H   24.311   -4.173   39.063
H   27.149   -4.807   38.290
H   24.002   -5.742   35.948
H   24.623   -4.361   35.037
H   23.827   -4.187   36.678
H   27.809   -5.523   36.227
H   27.009   -5.615   34.752
H   25.890   -7.659   36.470
H   27.643   -7.809   36.419
H   27.037   -8.984   34.716
H   27.513   -7.526   33.931
H   24.737   -7.390   34.695
H   24.907   -9.581   35.138
H   25.016   -9.919   33.342
H   23.520   -9.077   34.136
H   25.049   -8.106   31.922
H   26.291   -6.864   32.312
H   23.918   -5.799   33.323
H   23.601   -6.450   31.724
H   26.070   -4.801   31.921
H   24.720   -3.798   32.483
H   24.410   -5.441   29.968
H   23.084   -2.777   30.279
H   22.591   -4.064   31.266
H   22.607   -4.119   29.473
H   26.328   -3.216   30.309
H   24.992   -2.752   29.290
H   26.244   -5.560   28.694
H   27.413   -4.339   28.782
H   27.020   -3.941   26.589
H   25.590   -3.148   27.146
H   25.383   -6.085   26.847
H   24.764   -5.300   24.223
H   26.274   -6.171   24.709
H   26.274   -4.427   24.472
H   23.452   -3.999   27.053
H   23.265   -5.727   26.876
H   23.338   -4.588   25.433

