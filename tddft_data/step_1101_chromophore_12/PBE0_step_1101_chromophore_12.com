%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1101_chromophore_12 TDDFT with PBE1PBE functional

0 1
Mg   47.245   16.310   28.454
C   45.064   15.593   31.034
C   48.658   18.691   30.315
C   49.101   17.263   25.713
C   45.165   14.463   26.345
N   46.941   16.949   30.430
C   46.085   16.402   31.419
C   46.389   16.880   32.863
C   47.236   18.149   32.435
C   47.727   17.911   30.981
C   46.340   19.435   32.527
C   47.193   15.806   33.608
C   47.009   15.735   35.174
C   48.252   15.744   36.075
O   49.217   14.980   35.872
O   48.161   16.679   37.050
N   48.843   17.542   28.126
C   49.237   18.510   29.076
C   50.212   19.386   28.498
C   50.421   18.963   27.203
C   49.439   17.825   26.933
C   50.839   20.574   29.174
C   51.485   19.435   26.143
O   51.659   18.933   25.054
C   52.441   20.565   26.387
N   47.204   15.963   26.356
C   48.099   16.390   25.428
C   47.754   15.827   24.035
C   46.496   15.024   24.228
C   46.278   15.136   25.735
C   47.748   16.868   22.927
C   46.663   13.527   23.735
C   47.461   12.569   24.624
N   45.476   15.195   28.517
C   44.673   14.453   27.661
C   43.518   13.904   28.319
C   43.676   14.285   29.665
C   44.892   15.056   29.719
C   42.453   12.994   27.726
C   43.123   14.127   31.041
O   42.085   13.580   31.472
C   43.971   14.977   31.906
C   43.192   16.062   32.495
O   42.582   16.943   31.916
O   43.266   15.929   33.817
C   42.916   17.125   34.618
C   49.416   16.738   37.819
C   49.531   18.014   38.584
C   50.651   18.546   39.103
C   52.096   17.985   38.976
C   50.686   19.916   39.780
C   51.191   21.100   38.933
C   52.595   21.690   39.318
C   53.256   22.502   38.142
C   54.725   22.820   38.417
C   52.590   23.851   37.829
C   52.282   24.266   36.406
C   53.496   25.225   35.953
C   54.306   24.556   34.825
C   55.820   24.763   35.006
C   53.828   25.074   33.410
C   54.208   24.089   32.289
C   53.048   23.446   31.606
C   52.217   24.327   30.714
C   50.696   24.216   30.935
C   52.571   24.113   29.236
H   49.132   19.491   30.888
H   49.721   17.557   24.863
H   44.350   14.126   25.701
H   45.511   17.154   33.449
H   48.129   18.191   33.059
H   45.281   19.181   32.587
H   46.541   19.886   31.555
H   46.681   20.066   33.347
H   48.250   15.902   33.359
H   46.873   14.838   33.222
H   46.571   14.744   35.291
H   46.244   16.419   35.543
H   51.812   20.240   29.535
H   50.242   21.022   29.968
H   51.067   21.468   28.593
H   51.937   21.494   26.123
H   53.313   20.418   25.749
H   52.846   20.612   27.397
H   48.547   15.182   23.658
H   45.635   15.473   23.734
H   47.718   17.801   23.490
H   46.957   16.833   22.178
H   48.673   16.769   22.360
H   47.082   13.655   22.737
H   45.662   13.109   23.622
H   48.318   12.225   24.045
H   46.871   11.727   24.988
H   47.901   13.122   25.454
H   42.151   13.531   26.826
H   41.589   12.782   28.357
H   42.849   12.058   27.334
H   44.399   14.336   32.676
H   41.842   17.312   34.584
H   43.395   18.034   34.254
H   43.232   16.871   35.630
H   50.335   16.608   37.247
H   49.399   15.838   38.433
H   48.731   18.744   38.700
H   52.651   18.171   39.896
H   52.674   18.505   38.212
H   52.145   16.926   38.723
H   51.348   19.911   40.647
H   49.664   20.075   40.123
H   50.445   21.883   39.068
H   51.308   20.777   37.898
H   53.383   21.059   39.730
H   52.565   22.331   40.199
H   53.267   21.805   37.304
H   55.366   22.036   38.015
H   54.953   22.891   39.481
H   55.181   23.687   37.940
H   53.206   24.594   38.334
H   51.689   24.088   38.396
H   51.378   24.874   36.429
H   52.145   23.423   35.728
H   54.120   25.643   36.742
H   52.987   26.118   35.589
H   54.140   23.484   34.727
H   56.289   23.801   34.798
H   56.039   25.044   36.036
H   56.300   25.579   34.466
H   54.141   26.101   33.221
H   52.753   25.062   33.595
H   54.845   23.303   32.695
H   54.742   24.566   31.467
H   52.484   22.854   32.327
H   53.552   22.723   30.965
H   52.391   25.391   30.875
H   50.080   24.366   30.048
H   50.310   25.018   31.563
H   50.390   23.291   31.425
H   53.595   24.017   28.875
H   52.243   25.004   28.700
H   51.956   23.255   28.964

