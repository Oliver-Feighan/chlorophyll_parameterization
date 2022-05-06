%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_301_chromophore_25 TDDFT with blyp functional

0 1
Mg   -2.735   34.508   26.795
C   -3.457   32.639   29.647
C   -1.030   36.776   28.775
C   -2.310   36.527   24.034
C   -4.410   32.270   24.851
N   -2.437   34.708   28.959
C   -2.757   33.827   29.968
C   -2.241   34.293   31.364
C   -1.604   35.690   31.055
C   -1.695   35.762   29.528
C   -2.152   36.888   31.801
C   -1.257   33.205   32.000
C   -1.736   32.595   33.345
C   -0.851   32.673   34.592
O   0.029   31.857   34.862
O   -1.244   33.811   35.251
N   -1.754   36.388   26.526
C   -1.038   37.117   27.413
C   -0.359   38.196   26.751
C   -0.748   38.178   25.413
C   -1.633   36.994   25.257
C   0.659   39.042   27.472
C   -0.439   39.185   24.299
O   -0.820   38.992   23.162
C   0.227   40.497   24.577
N   -3.342   34.471   24.752
C   -3.040   35.371   23.898
C   -3.476   34.879   22.476
C   -3.837   33.395   22.639
C   -3.793   33.293   24.140
C   -4.626   35.797   21.959
C   -2.985   32.357   21.911
C   -3.365   30.871   21.823
N   -3.810   32.838   27.151
C   -4.441   31.972   26.246
C   -5.042   30.862   26.992
C   -4.640   31.018   28.311
C   -3.886   32.248   28.373
C   -5.867   29.832   26.415
C   -4.709   30.457   29.604
O   -5.277   29.470   30.015
C   -4.043   31.534   30.548
C   -5.105   32.108   31.470
O   -5.950   32.936   31.132
O   -4.989   31.531   32.696
C   -5.916   32.020   33.753
C   -0.508   33.901   36.513
C   -0.947   35.018   37.337
C   -1.911   35.081   38.288
C   -2.760   33.875   38.642
C   -2.126   36.273   39.240
C   -1.416   35.986   40.543
C   -1.300   37.271   41.414
C   -1.876   37.047   42.851
C   -3.426   37.283   42.812
C   -1.133   37.960   43.870
C   -0.923   37.287   45.233
C   0.535   37.397   45.799
C   1.358   36.097   45.705
C   0.899   34.971   46.615
C   2.849   36.304   45.953
C   3.674   35.260   45.257
C   4.473   35.673   43.957
C   4.410   34.600   42.846
C   5.498   34.855   41.823
C   3.036   34.512   42.114
H   -0.527   37.460   29.463
H   -2.134   37.120   23.134
H   -4.891   31.547   24.188
H   -3.069   34.495   32.044
H   -0.549   35.506   31.261
H   -2.809   36.538   32.598
H   -2.529   37.684   31.159
H   -1.308   37.395   32.268
H   -0.274   33.645   32.167
H   -1.099   32.417   31.263
H   -1.892   31.527   33.192
H   -2.683   33.092   33.553
H   0.710   38.674   28.497
H   0.475   40.115   27.426
H   1.545   38.847   26.868
H   1.312   40.468   24.672
H   -0.270   40.793   25.501
H   -0.079   41.245   23.846
H   -2.701   35.123   21.749
H   -4.881   33.216   22.383
H   -5.311   36.214   22.697
H   -5.312   35.081   21.506
H   -4.344   36.485   21.162
H   -1.983   32.349   22.341
H   -2.804   32.537   20.852
H   -4.337   30.739   22.298
H   -2.578   30.329   22.347
H   -3.365   30.508   20.796
H   -6.028   30.168   25.391
H   -6.779   29.829   27.012
H   -5.429   28.837   26.492
H   -3.228   31.064   31.098
H   -6.667   32.751   33.454
H   -5.395   32.654   34.471
H   -6.469   31.168   34.149
H   0.546   33.973   36.244
H   -0.658   32.988   37.090
H   -0.159   35.754   37.497
H   -3.050   33.813   39.691
H   -2.208   32.950   38.474
H   -3.671   33.971   38.051
H   -3.158   36.471   39.530
H   -1.769   37.208   38.808
H   -0.415   35.596   40.359
H   -2.028   35.265   41.084
H   -1.831   38.091   40.930
H   -0.238   37.490   41.527
H   -1.774   35.999   43.134
H   -3.688   38.330   42.969
H   -3.933   36.702   43.581
H   -3.722   37.091   41.781
H   -1.732   38.840   44.101
H   -0.207   38.304   43.410
H   -1.262   36.253   45.182
H   -1.570   37.757   45.973
H   0.432   37.661   46.851
H   1.050   38.160   45.216
H   1.210   35.569   44.763
H   1.618   34.665   47.374
H   0.686   34.058   46.059
H   -0.063   35.201   47.072
H   3.096   36.363   47.013
H   3.112   37.222   45.427
H   3.021   34.416   45.036
H   4.486   35.009   45.939
H   5.536   35.735   44.190
H   4.170   36.612   43.495
H   4.626   33.674   43.378
H   6.345   34.260   42.165
H   5.819   35.896   41.783
H   5.239   34.587   40.799
H   2.577   33.548   42.333
H   3.120   34.797   41.065
H   2.322   35.217   42.540

