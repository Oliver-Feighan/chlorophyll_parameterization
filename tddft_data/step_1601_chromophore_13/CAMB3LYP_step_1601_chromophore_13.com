%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1601_chromophore_13 TDDFT with cam-b3lyp functional

0 1
Mg   47.437   24.614   28.949
C   48.102   26.797   31.529
C   46.387   22.256   31.145
C   47.903   22.193   26.591
C   49.182   26.877   26.852
N   47.116   24.600   31.010
C   47.570   25.572   31.917
C   47.472   25.138   33.404
C   47.063   23.646   33.255
C   46.737   23.474   31.678
C   48.176   22.504   33.759
C   46.475   26.038   34.208
C   46.723   25.910   35.752
C   45.588   26.465   36.658
O   44.622   27.094   36.248
O   45.749   26.149   38.013
N   46.822   22.529   28.821
C   46.363   21.797   29.832
C   45.967   20.509   29.283
C   46.378   20.498   27.966
C   47.075   21.743   27.709
C   45.166   19.424   30.046
C   46.325   19.378   26.923
O   46.756   19.647   25.823
C   45.768   18.031   27.269
N   48.531   24.574   27.127
C   48.473   23.451   26.318
C   49.399   23.623   25.153
C   49.439   25.213   25.022
C   48.972   25.637   26.354
C   50.777   22.936   25.226
C   48.671   25.924   23.892
C   47.182   26.357   24.196
N   48.218   26.544   29.100
C   48.793   27.320   28.133
C   49.021   28.663   28.637
C   48.821   28.446   30.036
C   48.418   27.178   30.244
C   49.408   29.870   27.753
C   49.026   29.048   31.383
O   49.343   30.175   31.716
C   48.579   27.977   32.384
C   49.712   27.658   33.320
O   50.605   26.833   33.080
O   49.555   28.397   34.475
C   50.575   28.037   35.486
C   44.551   26.309   38.870
C   44.664   25.041   39.733
C   44.302   24.953   41.014
C   43.679   26.047   41.899
C   44.433   23.687   41.848
C   43.971   22.389   41.134
C   43.219   21.320   42.083
C   41.729   21.030   41.858
C   41.571   20.401   40.484
C   40.872   22.378   42.030
C   40.109   22.502   43.397
C   40.362   23.758   44.140
C   40.446   23.628   45.704
C   39.194   24.157   46.508
C   41.743   24.272   46.277
C   42.878   23.236   46.508
C   43.100   23.068   48.049
C   43.771   24.354   48.783
C   44.954   23.880   49.623
C   42.777   25.244   49.433
H   46.077   21.468   31.835
H   48.078   21.478   25.784
H   49.617   27.564   26.123
H   48.466   25.016   33.835
H   46.244   23.464   33.951
H   47.793   21.694   34.380
H   49.067   22.995   34.151
H   48.522   22.120   32.800
H   45.502   25.592   33.999
H   46.462   27.065   33.843
H   47.658   26.384   36.050
H   46.768   24.853   36.015
H   44.917   19.757   31.053
H   45.773   18.519   30.003
H   44.273   19.208   29.460
H   44.707   18.179   27.473
H   46.363   17.715   28.126
H   45.940   17.398   26.399
H   48.869   23.224   24.288
H   50.434   25.643   24.903
H   51.596   23.585   25.535
H   51.012   22.411   24.300
H   50.788   22.183   26.014
H   48.760   25.360   22.963
H   49.133   26.908   23.823
H   46.633   26.168   23.273
H   47.020   27.395   24.487
H   46.655   25.797   24.968
H   48.948   29.669   26.785
H   50.491   29.839   27.632
H   49.068   30.860   28.057
H   47.842   28.575   32.919
H   50.670   28.897   36.149
H   51.514   27.634   35.106
H   50.133   27.269   36.121
H   43.546   26.310   38.447
H   44.652   27.272   39.371
H   45.095   24.179   39.224
H   44.427   26.222   42.672
H   42.678   25.840   42.276
H   43.550   26.973   41.338
H   43.843   23.771   42.761
H   45.491   23.646   42.109
H   44.808   21.891   40.647
H   43.388   22.567   40.231
H   43.313   21.661   43.114
H   43.688   20.336   42.054
H   41.373   20.271   42.555
H   42.377   19.692   40.295
H   41.583   21.123   39.667
H   40.643   19.832   40.534
H   40.048   22.489   41.325
H   41.443   23.306   42.015
H   40.474   21.596   43.882
H   39.056   22.378   43.144
H   39.507   24.404   43.937
H   41.187   24.354   43.750
H   40.513   22.550   45.850
H   39.045   23.610   47.439
H   38.264   23.991   45.963
H   39.209   25.236   46.657
H   41.502   24.872   47.154
H   42.142   25.029   45.602
H   43.759   23.503   45.923
H   42.722   22.213   46.166
H   43.546   22.098   48.265
H   42.129   23.010   48.541
H   44.289   24.919   48.008
H   45.223   22.837   49.454
H   44.746   23.898   50.693
H   45.792   24.567   49.505
H   42.590   26.188   48.920
H   43.042   25.440   50.472
H   41.732   24.937   49.389

