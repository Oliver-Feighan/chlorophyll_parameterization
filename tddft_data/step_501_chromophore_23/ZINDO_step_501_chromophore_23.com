%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_501_chromophore_23 ZINDO

0 1
Mg   -10.089   40.792   42.947
C   -8.616   37.873   41.739
C   -7.782   42.641   41.174
C   -11.691   43.406   44.117
C   -13.011   38.808   43.846
N   -8.298   40.231   41.615
C   -7.899   38.981   41.233
C   -6.678   39.048   40.386
C   -6.440   40.596   40.172
C   -7.577   41.257   41.045
C   -4.986   41.069   40.423
C   -6.832   38.299   39.040
C   -5.676   37.323   38.828
C   -4.885   37.425   37.489
O   -3.848   38.095   37.420
O   -5.495   36.664   36.515
N   -9.754   42.766   42.703
C   -8.729   43.366   41.973
C   -8.872   44.797   42.208
C   -10.006   45.055   43.110
C   -10.525   43.687   43.427
C   -8.103   45.788   41.488
C   -10.495   46.415   43.392
O   -9.976   47.428   42.972
C   -11.745   46.708   44.146
N   -12.017   41.036   43.776
C   -12.309   42.201   44.439
C   -13.681   42.110   45.073
C   -14.242   40.842   44.547
C   -13.083   40.198   43.966
C   -13.708   42.291   46.646
C   -15.267   41.057   43.381
C   -16.322   40.059   43.244
N   -10.729   38.743   42.866
C   -11.880   38.083   43.316
C   -11.647   36.673   43.294
C   -10.421   36.526   42.656
C   -9.900   37.838   42.369
C   -12.608   35.666   43.809
C   -9.507   35.582   42.148
O   -9.550   34.372   42.157
C   -8.306   36.384   41.599
C   -7.079   35.935   42.276
O   -6.651   36.366   43.360
O   -6.556   34.851   41.609
C   -5.284   34.280   42.021
C   -4.682   36.339   35.380
C   -5.567   36.267   34.104
C   -5.543   37.135   33.130
C   -5.233   38.628   33.323
C   -6.294   36.838   31.818
C   -5.341   36.654   30.566
C   -5.948   36.493   29.221
C   -5.380   37.457   28.174
C   -4.524   36.743   27.077
C   -6.398   38.356   27.474
C   -6.769   39.601   28.387
C   -8.295   39.682   28.539
C   -8.751   39.910   29.948
C   -9.819   38.867   30.417
C   -9.411   41.355   30.060
C   -8.263   42.456   30.423
C   -8.563   43.732   29.612
C   -7.733   43.792   28.331
C   -6.670   44.855   28.246
C   -8.628   43.841   27.044
H   -6.984   43.159   40.639
H   -12.352   44.181   44.513
H   -13.898   38.251   44.155
H   -5.867   38.791   41.068
H   -6.604   40.924   39.146
H   -4.425   40.136   40.357
H   -4.810   41.649   41.329
H   -4.661   41.714   39.607
H   -6.760   38.942   38.163
H   -7.799   37.798   39.081
H   -6.244   36.396   38.751
H   -5.062   37.350   39.727
H   -7.399   45.263   40.842
H   -7.461   46.428   42.094
H   -8.771   46.371   40.853
H   -11.592   46.043   44.996
H   -12.597   46.391   43.544
H   -11.786   47.753   44.454
H   -14.192   42.985   44.673
H   -14.608   40.109   45.266
H   -12.740   42.639   47.007
H   -14.110   41.442   47.200
H   -14.307   43.142   46.969
H   -14.724   41.146   42.440
H   -15.785   42.009   43.496
H   -17.155   40.343   43.888
H   -15.968   39.050   43.458
H   -16.801   40.164   42.270
H   -13.094   36.015   44.720
H   -12.173   34.679   43.967
H   -13.399   35.663   43.059
H   -8.067   36.135   40.565
H   -5.235   33.707   42.947
H   -4.523   35.060   42.034
H   -5.072   33.535   41.255
H   -4.205   35.369   35.520
H   -3.841   37.023   35.267
H   -5.670   35.191   33.961
H   -4.896   38.856   34.335
H   -5.984   39.349   33.002
H   -4.357   38.859   32.719
H   -6.969   37.642   31.527
H   -6.892   35.932   31.912
H   -4.817   35.739   30.841
H   -4.796   37.598   30.549
H   -7.028   36.642   29.237
H   -5.650   35.473   28.977
H   -4.667   38.105   28.685
H   -4.042   37.441   26.393
H   -5.239   36.297   26.386
H   -3.773   35.997   27.337
H   -7.283   38.030   26.929
H   -5.809   38.820   26.683
H   -6.327   40.493   27.942
H   -6.346   39.375   29.365
H   -8.818   38.863   28.045
H   -8.631   40.524   27.935
H   -8.000   39.870   30.737
H   -9.642   38.391   31.381
H   -9.954   38.099   29.655
H   -10.827   39.277   30.352
H   -10.089   41.284   30.910
H   -9.994   41.727   29.218
H   -7.293   42.028   30.172
H   -8.069   42.678   31.473
H   -8.354   44.647   30.168
H   -9.619   43.847   29.371
H   -7.174   42.860   28.240
H   -6.263   45.228   29.186
H   -7.177   45.736   27.852
H   -5.879   44.610   27.538
H   -8.580   42.818   26.671
H   -8.350   44.633   26.349
H   -9.667   44.073   27.280

