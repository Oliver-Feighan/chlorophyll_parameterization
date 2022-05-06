%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_751_chromophore_16 TDDFT with PBE1PBE functional

0 1
Mg   41.220   41.164   27.361
C   40.226   43.642   29.657
C   41.808   39.018   29.903
C   42.342   38.920   25.033
C   40.713   43.510   24.895
N   41.105   41.331   29.578
C   40.614   42.453   30.286
C   40.591   42.123   31.777
C   41.083   40.599   31.784
C   41.340   40.249   30.356
C   42.408   40.379   32.640
C   39.184   42.157   32.486
C   38.755   41.484   33.799
C   38.098   42.297   34.901
O   36.869   42.407   34.963
O   39.050   42.771   35.790
N   42.035   39.279   27.444
C   42.122   38.608   28.619
C   42.536   37.259   28.263
C   42.801   37.231   26.910
C   42.386   38.448   26.379
C   42.633   36.103   29.242
C   43.338   36.081   26.091
O   43.505   36.151   24.843
C   43.732   34.837   26.756
N   41.368   41.138   25.302
C   41.888   40.139   24.545
C   41.677   40.431   23.038
C   41.242   41.854   23.002
C   41.072   42.212   24.485
C   43.036   40.107   22.231
C   40.002   42.152   22.078
C   38.678   41.465   22.467
N   40.596   43.212   27.182
C   40.479   44.050   26.126
C   40.067   45.373   26.544
C   39.916   45.224   27.950
C   40.283   43.926   28.245
C   39.807   46.636   25.764
C   39.479   45.944   29.137
O   39.155   47.059   29.377
C   39.559   44.880   30.269
C   40.394   45.380   31.330
O   41.575   45.540   31.280
O   39.609   45.750   32.423
C   40.184   46.269   33.679
C   38.444   43.469   36.977
C   38.139   42.504   38.100
C   38.249   42.694   39.446
C   39.002   43.857   40.062
C   37.665   41.592   40.403
C   38.685   40.627   41.069
C   37.987   39.359   41.648
C   38.118   39.165   43.196
C   39.561   38.691   43.520
C   37.083   38.249   43.758
C   36.970   38.188   45.324
C   37.085   36.662   45.744
C   35.631   36.136   46.208
C   35.514   36.165   47.755
C   35.270   34.742   45.551
C   34.231   34.830   44.413
C   34.558   34.055   43.138
C   34.413   34.880   41.791
C   35.777   35.218   41.117
C   33.465   34.205   40.770
H   41.851   38.269   30.697
H   42.798   38.331   24.234
H   40.478   44.216   24.097
H   41.268   42.790   32.311
H   40.297   39.979   32.215
H   43.273   40.145   32.019
H   42.231   39.566   33.345
H   42.704   41.273   33.188
H   38.360   42.112   31.774
H   39.166   43.215   32.748
H   39.765   41.218   34.110
H   38.066   40.647   33.680
H   42.016   35.212   29.128
H   42.277   36.370   30.237
H   43.648   35.724   29.363
H   44.585   35.065   27.395
H   44.035   34.165   25.953
H   42.931   34.245   27.200
H   40.931   39.724   22.674
H   42.087   42.446   22.649
H   43.437   40.918   21.624
H   42.769   39.380   21.463
H   43.851   39.806   22.890
H   40.321   41.721   21.129
H   39.804   43.220   21.992
H   37.932   42.244   22.312
H   38.737   40.979   23.441
H   38.497   40.739   21.674
H   39.520   46.361   24.749
H   40.694   47.261   25.666
H   38.930   47.184   26.109
H   38.570   44.849   30.727
H   41.224   45.943   33.714
H   39.759   45.876   34.602
H   40.135   47.356   33.619
H   37.523   43.979   36.694
H   39.195   44.208   37.258
H   37.662   41.616   37.683
H   38.711   44.027   41.098
H   38.771   44.783   39.536
H   40.086   43.776   40.146
H   37.031   40.946   39.796
H   37.200   42.138   41.225
H   39.102   41.036   41.990
H   39.499   40.339   40.404
H   38.380   38.586   40.987
H   36.935   39.545   41.434
H   38.127   40.136   43.692
H   39.979   39.337   44.292
H   40.198   38.648   42.636
H   39.490   37.692   43.951
H   37.113   37.285   43.249
H   36.158   38.746   43.467
H   35.968   38.599   45.451
H   37.645   38.891   45.813
H   37.816   36.534   46.543
H   37.397   36.069   44.884
H   34.915   36.866   45.832
H   34.509   36.287   48.161
H   36.057   37.058   48.063
H   36.078   35.319   48.147
H   35.059   34.041   46.358
H   36.231   34.323   45.253
H   33.885   35.835   44.172
H   33.354   34.328   44.820
H   33.888   33.196   43.101
H   35.579   33.707   43.294
H   33.899   35.821   41.988
H   35.727   36.277   40.864
H   36.012   34.617   40.238
H   36.617   35.076   41.796
H   32.546   34.789   40.731
H   33.187   33.191   41.059
H   33.952   34.051   39.807

