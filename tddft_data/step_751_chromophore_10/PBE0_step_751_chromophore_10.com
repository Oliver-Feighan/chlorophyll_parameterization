%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_751_chromophore_10 TDDFT with PBE1PBE functional

0 1
Mg   41.426   8.005   28.735
C   43.006   9.343   31.598
C   39.129   6.664   30.651
C   40.159   6.651   25.888
C   44.045   9.370   26.856
N   41.167   8.026   30.921
C   41.822   8.652   31.883
C   41.269   8.284   33.232
C   40.100   7.195   32.905
C   40.121   7.287   31.425
C   40.208   5.708   33.507
C   40.705   9.521   33.967
C   40.998   9.735   35.503
C   41.535   11.125   35.979
O   42.076   11.949   35.233
O   41.083   11.408   37.307
N   39.813   6.915   28.358
C   38.917   6.517   29.303
C   37.819   5.808   28.704
C   38.057   5.871   27.286
C   39.402   6.425   27.089
C   36.744   5.040   29.500
C   37.079   5.451   26.215
O   37.453   5.537   25.024
C   35.706   4.858   26.533
N   42.065   7.975   26.660
C   41.420   7.315   25.686
C   42.102   7.368   24.330
C   43.245   8.389   24.610
C   43.108   8.652   26.146
C   42.671   6.036   23.850
C   43.115   9.612   23.743
C   42.094   10.686   24.165
N   43.138   9.241   29.111
C   44.036   9.749   28.221
C   45.120   10.432   28.848
C   44.724   10.396   30.243
C   43.567   9.551   30.312
C   46.342   11.072   28.162
C   45.065   10.756   31.640
O   46.025   11.361   32.140
C   43.947   10.170   32.526
C   44.571   9.136   33.445
O   45.520   8.394   33.147
O   43.984   9.148   34.701
C   44.465   8.219   35.708
C   41.450   12.751   37.685
C   40.770   12.873   39.003
C   41.373   12.714   40.236
C   42.793   12.187   40.442
C   40.589   12.810   41.533
C   39.686   11.568   41.739
C   40.130   10.642   42.908
C   39.148   10.741   44.160
C   39.845   10.244   45.419
C   37.683   10.114   43.971
C   36.525   10.750   44.819
C   35.766   11.831   43.981
C   34.354   11.452   43.499
C   34.217   11.769   42.000
C   33.235   12.138   44.388
C   33.182   11.671   45.837
C   33.895   12.592   46.936
C   34.794   11.896   47.893
C   33.852   11.105   48.750
C   35.676   12.880   48.676
H   38.253   6.369   31.232
H   39.779   6.255   24.944
H   44.858   9.887   26.343
H   42.079   7.748   33.728
H   39.109   7.497   33.242
H   39.285   5.573   34.071
H   41.026   5.786   34.223
H   40.366   4.878   32.819
H   39.617   9.492   33.901
H   40.982   10.404   33.391
H   41.687   8.947   35.808
H   40.048   9.468   35.965
H   35.908   5.726   29.639
H   37.014   4.794   30.527
H   36.323   4.128   29.078
H   35.301   5.568   27.255
H   35.858   3.811   26.795
H   35.171   4.990   25.593
H   41.369   7.718   23.604
H   44.243   7.978   24.458
H   42.754   6.062   22.763
H   42.049   5.193   24.151
H   43.668   5.861   24.256
H   42.903   9.262   22.733
H   44.131   10.002   23.679
H   42.705   11.575   24.320
H   41.701   10.682   25.182
H   41.283   10.823   23.450
H   46.391   10.886   27.089
H   47.239   10.708   28.663
H   46.402   12.160   28.189
H   43.339   10.833   33.141
H   45.472   7.811   35.628
H   43.694   7.450   35.764
H   44.387   8.734   36.665
H   40.970   13.484   37.037
H   42.515   12.839   37.901
H   39.764   13.283   38.910
H   43.614   12.813   40.091
H   42.753   11.332   39.768
H   42.993   11.818   41.448
H   39.858   13.616   41.607
H   41.248   12.931   42.393
H   39.493   11.008   40.824
H   38.701   11.995   41.925
H   41.157   10.777   43.249
H   40.189   9.595   42.610
H   38.784   11.760   44.287
H   39.231   10.462   46.293
H   40.757   10.818   45.581
H   40.019   9.168   45.433
H   37.566   9.094   44.337
H   37.520   10.029   42.897
H   37.001   11.173   45.704
H   35.831   9.943   45.057
H   36.367   12.038   43.095
H   35.673   12.761   44.541
H   34.274   10.366   43.545
H   33.170   11.722   41.700
H   34.690   10.955   41.452
H   34.723   12.702   41.751
H   32.241   12.137   43.941
H   33.517   13.189   44.323
H   33.556   10.651   45.921
H   32.130   11.668   46.123
H   33.209   13.239   47.481
H   34.446   13.275   46.289
H   35.436   11.230   47.316
H   34.275   10.103   48.688
H   32.804   10.968   48.484
H   33.926   11.322   49.816
H   36.677   12.459   48.580
H   35.483   12.884   49.749
H   35.734   13.945   48.451

