%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1201_chromophore_20 TDDFT with blyp functional

0 1
Mg   7.088   56.847   41.817
C   5.914   53.635   41.432
C   10.185   55.638   40.733
C   8.119   60.046   41.206
C   3.823   57.967   42.391
N   7.880   54.899   41.145
C   7.207   53.713   40.997
C   8.045   52.590   40.504
C   9.481   53.197   40.722
C   9.200   54.698   40.944
C   10.300   52.449   41.828
C   7.791   52.222   39.048
C   8.288   50.851   38.595
C   7.530   50.076   37.486
O   7.226   48.862   37.470
O   7.068   50.944   36.548
N   8.835   57.768   41.036
C   10.010   57.074   40.639
C   10.900   58.011   39.931
C   10.333   59.322   39.987
C   9.028   59.111   40.760
C   12.103   57.528   39.263
C   10.861   60.553   39.337
O   11.925   60.562   38.773
C   10.080   61.826   39.225
N   6.069   58.697   41.722
C   6.768   59.847   41.644
C   5.913   61.086   42.007
C   4.509   60.422   42.427
C   4.790   58.958   42.071
C   6.513   61.988   43.170
C   3.277   61.004   41.683
C   2.016   61.045   42.594
N   5.214   55.975   42.076
C   3.966   56.575   42.337
C   3.011   55.432   42.509
C   3.756   54.321   42.155
C   5.114   54.667   41.919
C   1.564   55.573   42.681
C   3.613   52.924   42.037
O   2.614   52.268   42.302
C   4.989   52.334   41.570
C   4.822   51.510   40.381
O   4.303   51.908   39.365
O   5.174   50.191   40.623
C   4.662   49.339   39.514
C   6.316   50.505   35.402
C   6.549   51.413   34.275
C   6.545   51.205   32.917
C   5.809   49.995   32.418
C   7.233   52.161   31.988
C   6.462   53.465   31.911
C   5.946   53.736   30.470
C   6.301   55.106   29.922
C   5.269   55.396   28.785
C   7.778   55.199   29.407
C   8.434   56.509   29.810
C   9.216   56.295   31.111
C   10.687   55.897   30.747
C   11.098   54.474   31.246
C   11.803   56.885   31.186
C   12.111   57.890   30.055
C   13.557   57.862   29.610
C   13.712   57.758   28.078
C   13.919   56.282   27.585
C   14.809   58.553   27.434
H   11.135   55.163   40.480
H   8.308   61.118   41.111
H   2.786   58.227   42.612
H   7.879   51.702   41.115
H   10.113   53.144   39.835
H   10.689   53.250   42.458
H   11.160   51.876   41.482
H   9.563   51.798   42.298
H   8.405   52.911   38.468
H   6.778   52.441   38.710
H   8.464   50.227   39.472
H   9.268   50.980   38.136
H   12.006   57.755   38.202
H   12.277   56.454   39.330
H   13.069   57.765   39.708
H   9.719   62.143   40.203
H   9.346   61.840   38.420
H   10.875   62.533   38.986
H   5.908   61.443   40.976
H   4.323   60.469   43.500
H   5.661   62.334   43.756
H   7.122   62.834   42.852
H   7.186   61.370   43.764
H   3.098   60.288   40.882
H   3.511   61.996   41.296
H   1.615   60.045   42.429
H   1.255   61.727   42.214
H   2.149   61.184   43.667
H   1.176   55.751   41.678
H   1.137   56.394   43.257
H   1.118   54.633   43.005
H   5.480   51.690   42.299
H   3.577   49.358   39.612
H   5.076   48.339   39.639
H   5.028   49.715   38.559
H   5.263   50.427   35.670
H   6.657   49.512   35.107
H   7.132   52.275   34.602
H   6.282   49.363   31.666
H   4.777   50.241   32.170
H   5.592   49.243   33.177
H   7.211   51.542   31.091
H   8.287   52.316   32.218
H   7.166   54.232   32.237
H   5.609   53.418   32.587
H   4.858   53.685   30.504
H   6.383   53.003   29.791
H   6.018   55.925   30.582
H   4.264   55.376   29.206
H   5.398   54.676   27.977
H   5.452   56.419   28.458
H   7.804   55.090   28.322
H   8.315   54.386   29.895
H   7.735   57.330   29.968
H   9.084   56.900   29.027
H   8.858   55.564   31.836
H   9.291   57.318   31.478
H   10.674   55.779   29.663
H   10.231   53.946   31.643
H   11.866   54.549   32.016
H   11.423   53.825   30.433
H   12.683   56.308   31.468
H   11.500   57.499   32.035
H   12.018   58.879   30.504
H   11.465   57.841   29.179
H   14.048   56.940   29.920
H   14.073   58.765   29.938
H   12.796   58.105   27.601
H   13.562   56.092   26.573
H   13.484   55.591   28.307
H   14.979   56.032   27.563
H   15.242   58.030   26.581
H   15.553   58.755   28.204
H   14.360   59.476   27.066

