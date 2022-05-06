%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_401_chromophore_12 TDDFT with cam-b3lyp functional

0 1
Mg   47.501   14.766   27.522
C   45.534   14.866   30.495
C   49.486   17.280   28.811
C   49.107   14.843   24.610
C   45.060   12.647   26.228
N   47.484   15.876   29.423
C   46.718   15.680   30.586
C   47.231   16.407   31.779
C   48.448   17.283   31.232
C   48.480   16.857   29.718
C   48.392   18.830   31.495
C   47.577   15.447   32.864
C   47.314   15.940   34.267
C   48.384   15.711   35.360
O   49.134   14.761   35.483
O   48.418   16.779   36.283
N   49.163   15.905   26.790
C   49.790   16.933   27.444
C   50.754   17.574   26.532
C   50.811   16.657   25.349
C   49.666   15.757   25.521
C   51.618   18.822   26.831
C   51.762   16.767   24.152
O   51.670   16.017   23.149
C   52.927   17.794   24.157
N   47.114   13.858   25.697
C   47.926   14.022   24.657
C   47.571   13.060   23.511
C   46.360   12.259   24.090
C   46.177   12.894   25.448
C   47.216   13.842   22.164
C   46.717   10.666   24.244
C   47.289   9.953   22.990
N   45.556   13.997   28.148
C   44.744   13.148   27.471
C   43.545   12.972   28.331
C   43.893   13.566   29.540
C   45.121   14.189   29.357
C   42.288   12.154   27.992
C   43.367   13.843   30.857
O   42.312   13.534   31.370
C   44.383   14.680   31.554
C   43.771   15.960   31.949
O   43.481   16.871   31.235
O   43.560   15.919   33.277
C   42.817   16.993   33.904
C   49.418   16.626   37.373
C   49.325   17.869   38.339
C   50.263   18.298   39.201
C   51.521   17.601   39.594
C   50.041   19.494   40.031
C   50.736   20.818   39.464
C   51.927   21.348   40.268
C   53.141   21.870   39.368
C   54.418   22.165   40.254
C   52.701   23.228   38.685
C   52.937   22.987   37.229
C   53.054   24.325   36.428
C   53.508   24.272   34.987
C   54.995   24.921   34.887
C   52.562   25.015   33.980
C   52.555   24.443   32.520
C   51.124   24.224   31.913
C   51.182   23.639   30.544
C   50.376   24.356   29.556
C   50.877   22.093   30.589
H   50.150   18.041   29.226
H   49.719   14.650   23.726
H   44.395   11.950   25.713
H   46.452   17.061   32.171
H   49.348   16.798   31.610
H   48.499   19.379   30.560
H   49.208   19.035   32.188
H   47.480   19.223   31.945
H   48.614   15.135   32.744
H   47.126   14.456   32.807
H   46.377   15.534   34.649
H   47.188   17.022   34.295
H   51.265   19.593   26.145
H   52.705   18.782   26.762
H   51.509   19.222   27.839
H   53.262   17.765   23.120
H   53.721   17.373   24.773
H   52.481   18.760   24.394
H   48.464   12.500   23.234
H   45.478   12.531   23.511
H   47.923   14.556   21.742
H   46.219   14.273   22.251
H   47.011   13.131   21.364
H   45.811   10.216   24.649
H   47.401   10.339   25.027
H   46.750   9.067   22.655
H   48.268   9.513   23.176
H   47.235   10.618   22.128
H   42.562   11.190   27.563
H   41.846   12.701   27.160
H   41.658   12.154   28.881
H   44.855   14.189   32.405
H   43.461   17.423   34.672
H   41.877   16.684   34.360
H   42.660   17.845   33.243
H   50.405   16.471   36.937
H   49.091   15.730   37.901
H   48.451   18.500   38.179
H   51.844   16.830   38.895
H   51.299   17.102   40.537
H   52.299   18.325   39.840
H   50.358   19.244   41.044
H   48.991   19.770   40.127
H   50.098   21.645   39.151
H   51.221   20.535   38.530
H   52.374   20.608   40.932
H   51.522   22.108   40.935
H   53.245   21.075   38.630
H   54.293   21.886   41.300
H   54.772   23.196   40.224
H   55.210   21.527   39.864
H   53.349   24.052   38.984
H   51.658   23.515   38.822
H   52.051   22.510   36.809
H   53.862   22.459   37.000
H   53.830   24.835   37.000
H   52.113   24.854   36.576
H   53.621   23.225   34.704
H   55.545   24.146   34.352
H   55.462   25.133   35.848
H   55.092   25.801   34.251
H   52.927   26.036   33.864
H   51.551   25.237   34.320
H   53.023   23.462   32.425
H   53.181   25.044   31.861
H   50.535   25.141   31.927
H   50.568   23.569   32.583
H   52.221   23.684   30.217
H   49.953   23.649   28.842
H   50.851   25.262   29.178
H   49.468   24.775   29.991
H   51.034   21.707   31.596
H   51.518   21.518   29.922
H   49.857   21.892   30.261

