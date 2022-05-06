%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_601_chromophore_25 TDDFT with wB97XD functional

0 1
Mg   -2.355   34.495   27.303
C   -3.267   32.399   29.907
C   -0.502   36.359   29.400
C   -2.022   36.755   24.880
C   -4.199   32.480   25.056
N   -2.015   34.346   29.487
C   -2.390   33.369   30.329
C   -1.839   33.678   31.771
C   -0.857   34.966   31.538
C   -1.164   35.304   30.065
C   -1.076   36.099   32.551
C   -1.037   32.529   32.242
C   -1.485   31.932   33.520
C   -0.385   32.031   34.630
O   0.396   31.139   34.868
O   -0.455   33.224   35.281
N   -1.464   36.305   27.202
C   -0.754   36.950   28.141
C   0.001   38.090   27.629
C   -0.609   38.350   26.336
C   -1.404   37.050   26.065
C   0.963   38.895   28.479
C   -0.428   39.545   25.314
O   -0.888   39.500   24.209
C   0.298   40.752   25.763
N   -2.978   34.580   25.263
C   -2.754   35.663   24.504
C   -3.364   35.463   23.099
C   -3.666   33.974   23.064
C   -3.644   33.634   24.572
C   -4.598   36.353   22.867
C   -2.626   33.099   22.186
C   -3.290   32.243   21.113
N   -3.446   32.709   27.439
C   -4.175   32.048   26.456
C   -4.895   30.945   27.018
C   -4.582   30.998   28.452
C   -3.717   32.114   28.599
C   -5.696   29.879   26.290
C   -4.839   30.484   29.756
O   -5.620   29.581   30.074
C   -3.981   31.355   30.722
C   -4.914   31.975   31.755
O   -5.596   32.989   31.617
O   -4.757   31.313   32.940
C   -5.570   31.757   34.052
C   0.442   33.279   36.506
C   -0.276   34.041   37.604
C   0.246   34.531   38.719
C   1.650   34.370   39.198
C   -0.615   35.564   39.528
C   -0.663   35.267   41.035
C   0.072   36.286   41.875
C   -0.528   36.666   43.268
C   -1.500   37.809   43.035
C   0.525   36.991   44.309
C   0.266   36.446   45.687
C   1.546   36.743   46.610
C   2.383   35.482   46.883
C   1.974   34.793   48.284
C   3.878   35.736   46.913
C   4.712   34.605   46.357
C   5.617   34.977   45.162
C   5.391   34.152   43.918
C   6.640   33.738   43.115
C   4.389   34.847   42.902
H   0.251   36.649   30.137
H   -1.956   37.497   24.082
H   -4.656   31.736   24.400
H   -2.686   33.965   32.395
H   0.216   34.772   31.518
H   -0.147   36.222   33.107
H   -1.821   35.645   33.205
H   -1.325   37.013   32.012
H   0.018   32.798   32.188
H   -1.098   31.740   31.492
H   -1.703   30.880   33.338
H   -2.366   32.485   33.848
H   1.483   38.236   29.174
H   0.479   39.677   29.064
H   1.592   39.369   27.725
H   0.183   41.479   24.959
H   1.382   40.760   25.883
H   -0.016   41.181   26.714
H   -2.642   35.735   22.328
H   -4.646   33.699   22.675
H   -5.112   36.481   23.820
H   -5.242   35.855   22.142
H   -4.465   37.253   22.266
H   -2.144   32.417   22.886
H   -1.791   33.701   21.828
H   -4.123   32.844   20.751
H   -3.553   31.257   21.498
H   -2.694   32.031   20.226
H   -5.270   28.888   26.442
H   -5.674   30.033   25.212
H   -6.697   29.818   26.717
H   -3.261   30.730   31.250
H   -4.925   32.282   34.755
H   -5.840   30.804   34.507
H   -6.402   32.402   33.771
H   1.294   33.933   36.321
H   0.792   32.314   36.874
H   -1.290   34.369   37.372
H   2.166   35.279   38.887
H   2.036   33.371   38.994
H   1.687   34.561   40.270
H   -1.638   35.518   39.156
H   -0.181   36.523   39.244
H   -0.225   34.288   41.233
H   -1.729   35.208   41.257
H   0.389   37.180   41.337
H   1.001   35.809   42.187
H   -1.115   35.833   43.654
H   -2.038   38.101   43.937
H   -2.322   37.467   42.405
H   -1.004   38.681   42.609
H   0.416   38.063   44.474
H   1.507   36.698   43.937
H   0.095   35.391   45.472
H   -0.568   36.991   46.131
H   1.296   37.158   47.586
H   2.111   37.471   46.029
H   2.107   34.754   46.120
H   2.434   33.807   48.220
H   0.891   34.723   48.388
H   2.170   35.514   49.078
H   4.256   36.133   47.855
H   4.124   36.520   46.197
H   4.075   33.770   46.069
H   5.340   34.327   47.203
H   6.649   34.899   45.504
H   5.448   36.011   44.863
H   4.935   33.176   44.082
H   6.688   34.254   42.156
H   6.624   32.681   42.852
H   7.542   33.968   43.682
H   4.024   34.056   42.246
H   4.907   35.616   42.329
H   3.532   35.271   43.425

