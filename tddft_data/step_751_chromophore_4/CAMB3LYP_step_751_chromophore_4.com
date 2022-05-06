%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_751_chromophore_4 TDDFT with cam-b3lyp functional

0 1
Mg   8.924   3.230   28.323
C   10.213   1.947   31.296
C   7.260   5.652   30.233
C   7.294   4.333   25.633
C   9.933   0.483   26.611
N   8.668   3.659   30.610
C   9.382   3.011   31.599
C   9.112   3.670   32.974
C   8.084   4.815   32.569
C   7.996   4.750   31.060
C   6.675   4.513   33.146
C   10.414   4.237   33.711
C   10.086   4.841   35.109
C   10.867   4.396   36.327
O   12.110   4.406   36.327
O   10.128   3.831   37.380
N   7.604   4.845   28.008
C   7.114   5.803   28.838
C   6.326   6.790   28.172
C   6.230   6.323   26.809
C   7.115   5.157   26.775
C   5.646   7.984   28.834
C   5.587   6.945   25.577
O   5.616   6.470   24.449
C   4.867   8.297   25.700
N   8.659   2.540   26.374
C   8.035   3.175   25.430
C   8.093   2.409   24.102
C   8.850   1.118   24.456
C   9.144   1.316   25.931
C   6.708   1.995   23.587
C   10.112   0.727   23.618
C   9.803   -0.013   22.288
N   9.977   1.666   28.756
C   10.437   0.610   27.953
C   11.298   -0.256   28.662
C   11.258   0.235   29.966
C   10.477   1.381   29.964
C   11.894   -1.535   28.173
C   11.767   0.005   31.269
O   12.479   -0.933   31.678
C   11.132   1.112   32.213
C   10.304   0.538   33.409
O   9.206   0.039   33.338
O   11.116   0.531   34.563
C   10.548   -0.135   35.766
C   10.904   3.347   38.561
C   10.745   4.477   39.585
C   11.371   4.558   40.772
C   12.348   3.552   41.379
C   11.249   5.780   41.562
C   9.909   6.146   42.205
C   9.117   7.373   41.606
C   8.924   8.464   42.698
C   7.604   8.199   43.474
C   8.989   9.847   42.120
C   10.458   10.444   42.027
C   10.470   11.995   42.326
C   10.946   12.319   43.794
C   12.004   13.455   43.959
C   9.762   12.396   44.850
C   9.518   11.087   45.649
C   9.306   11.347   47.192
C   9.039   9.953   47.899
C   7.664   9.284   47.738
C   9.278   10.210   49.404
H   6.769   6.399   30.861
H   6.702   4.609   24.758
H   10.246   -0.437   26.113
H   8.603   2.824   33.434
H   8.302   5.813   32.948
H   6.715   3.705   33.875
H   6.005   4.257   32.325
H   6.360   5.346   33.775
H   10.810   5.009   33.052
H   11.083   3.376   33.694
H   9.024   4.671   35.281
H   10.342   5.900   35.067
H   4.608   7.716   29.031
H   5.570   8.828   28.149
H   6.078   8.354   29.764
H   4.558   8.634   24.710
H   5.570   8.993   26.157
H   4.031   8.224   26.395
H   8.530   3.119   23.400
H   8.147   0.300   24.293
H   5.908   1.866   24.317
H   6.709   1.210   22.831
H   6.358   2.859   23.021
H   10.867   0.178   24.181
H   10.668   1.638   23.394
H   9.904   0.628   21.413
H   8.764   -0.343   22.330
H   10.361   -0.928   22.087
H   11.113   -2.029   27.596
H   12.281   -2.241   28.908
H   12.687   -1.309   27.460
H   11.887   1.683   32.754
H   10.550   -1.207   35.572
H   9.502   0.061   36.003
H   11.092   0.040   36.694
H   11.937   3.125   38.294
H   10.329   2.467   38.849
H   10.292   5.429   39.306
H   12.045   3.260   42.384
H   13.340   4.002   41.422
H   12.388   2.742   40.650
H   11.446   6.649   40.934
H   11.977   5.702   42.370
H   10.104   6.238   43.273
H   9.295   5.246   42.155
H   8.147   7.061   41.218
H   9.626   7.928   40.818
H   9.767   8.316   43.373
H   7.388   7.134   43.389
H   6.776   8.687   42.961
H   7.734   8.470   44.522
H   8.213   10.440   42.603
H   8.673   9.738   41.082
H   11.008   10.282   41.100
H   11.056   9.955   42.796
H   9.439   12.307   42.160
H   10.993   12.423   41.470
H   11.498   11.461   44.176
H   13.027   13.080   43.926
H   11.895   13.818   44.981
H   11.776   14.295   43.303
H   8.926   12.602   44.181
H   9.798   13.225   45.557
H   10.292   10.337   45.487
H   8.594   10.571   45.387
H   8.465   11.997   47.434
H   10.260   11.801   47.464
H   9.783   9.207   47.620
H   6.990   9.764   47.029
H   7.281   8.797   48.635
H   7.930   8.454   47.083
H   8.394   9.966   49.994
H   9.581   11.219   49.682
H   10.170   9.614   49.598

