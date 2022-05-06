%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1751_chromophore_18 TDDFT with cam-b3lyp functional

0 1
Mg   35.432   49.389   25.371
C   35.397   47.375   28.289
C   33.955   51.856   27.180
C   35.042   51.057   22.542
C   36.297   46.487   23.567
N   34.803   49.595   27.543
C   34.974   48.723   28.572
C   34.551   49.312   29.929
C   34.017   50.707   29.515
C   34.148   50.691   27.967
C   32.612   51.137   30.041
C   35.729   49.345   30.895
C   35.317   48.824   32.291
C   33.825   48.959   32.772
O   33.213   47.946   33.005
O   33.311   50.222   33.008
N   34.730   51.287   24.907
C   34.182   52.160   25.793
C   33.873   53.429   25.102
C   33.938   53.093   23.795
C   34.578   51.734   23.664
C   33.609   54.714   25.779
C   33.423   53.896   22.584
O   33.626   53.608   21.368
C   32.769   55.229   22.871
N   35.774   48.848   23.315
C   35.616   49.765   22.317
C   35.950   49.125   20.967
C   36.250   47.681   21.298
C   36.185   47.644   22.853
C   34.873   49.260   19.868
C   37.588   47.096   20.719
C   38.838   47.737   21.431
N   35.765   47.288   25.791
C   36.163   46.230   24.950
C   36.413   45.021   25.743
C   36.170   45.494   27.067
C   35.755   46.847   27.032
C   36.735   43.698   25.179
C   36.187   44.989   28.447
O   36.527   43.912   28.892
C   35.625   46.231   29.307
C   34.357   45.728   30.030
O   33.147   46.061   29.857
O   34.779   44.908   31.062
C   33.664   44.407   31.882
C   31.905   50.164   33.493
C   31.489   51.582   33.682
C   31.214   52.188   34.900
C   31.663   51.494   36.202
C   30.780   53.700   35.021
C   32.036   54.635   35.120
C   31.631   56.021   34.643
C   32.002   57.243   35.682
C   33.530   57.212   35.903
C   31.352   58.543   35.212
C   30.428   59.108   36.276
C   31.153   59.799   37.436
C   30.400   61.083   37.873
C   31.296   62.194   38.261
C   29.284   60.726   38.914
C   28.004   60.324   38.112
C   26.863   61.332   38.272
C   27.107   62.677   37.472
C   27.575   63.809   38.359
C   26.121   63.110   36.356
H   33.480   52.663   27.742
H   34.709   51.583   21.644
H   36.535   45.546   23.067
H   33.742   48.662   30.263
H   34.754   51.419   29.886
H   32.008   50.418   30.595
H   32.027   51.355   29.148
H   32.675   52.078   30.588
H   36.189   50.325   31.020
H   36.471   48.630   30.539
H   35.944   49.371   32.995
H   35.741   47.822   32.365
H   32.550   54.962   25.701
H   34.210   55.558   25.441
H   34.018   54.679   26.788
H   31.847   55.002   23.407
H   32.327   55.520   21.918
H   33.323   55.993   23.417
H   36.925   49.521   20.683
H   35.451   46.997   21.014
H   35.100   50.140   19.265
H   33.966   49.405   20.455
H   34.819   48.309   19.339
H   37.674   47.328   19.657
H   37.614   46.006   20.699
H   38.803   47.609   22.513
H   38.768   48.817   21.297
H   39.749   47.386   20.947
H   35.794   43.241   24.876
H   37.197   43.110   25.973
H   37.404   43.831   24.329
H   36.320   46.502   30.103
H   33.115   43.667   31.298
H   32.947   45.159   32.211
H   34.135   43.940   32.747
H   31.656   49.623   34.406
H   31.270   49.727   32.723
H   30.985   51.997   32.809
H   30.903   51.185   36.920
H   32.222   52.301   36.675
H   32.373   50.674   36.097
H   30.298   53.746   35.997
H   30.201   53.912   34.122
H   32.775   54.142   34.487
H   32.493   54.731   36.105
H   30.571   56.090   34.398
H   32.127   56.079   33.674
H   31.762   56.916   36.694
H   33.965   58.207   35.995
H   34.116   56.712   35.132
H   33.683   56.614   36.801
H   30.822   58.424   34.267
H   32.132   59.297   35.111
H   29.859   58.287   36.712
H   29.696   59.649   35.675
H   32.214   59.972   37.256
H   31.180   59.201   38.347
H   29.872   61.443   36.990
H   31.364   62.855   37.397
H   32.304   61.793   38.365
H   31.080   62.610   39.246
H   29.102   61.607   39.529
H   29.606   59.826   39.439
H   27.621   59.400   38.546
H   28.240   60.189   37.057
H   26.672   61.510   39.331
H   25.982   60.943   37.762
H   27.956   62.472   36.819
H   26.699   64.418   38.578
H   28.427   64.279   37.867
H   27.882   63.372   39.309
H   25.502   62.276   36.027
H   26.672   63.339   35.444
H   25.494   63.938   36.686

