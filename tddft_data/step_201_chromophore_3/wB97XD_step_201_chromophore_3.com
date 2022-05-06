%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_201_chromophore_3 TDDFT with wB97XD functional

0 1
Mg   1.245   8.391   25.669
C   2.177   10.333   28.514
C   1.464   5.602   27.691
C   0.597   6.603   23.001
C   1.881   11.148   23.743
N   1.735   7.936   27.818
C   1.906   8.967   28.753
C   2.060   8.396   30.118
C   2.168   6.885   29.806
C   1.785   6.792   28.391
C   3.551   6.258   30.072
C   0.908   8.891   31.154
C   1.394   9.133   32.617
C   2.642   8.327   33.034
O   3.771   8.795   33.222
O   2.255   7.089   33.391
N   1.091   6.322   25.354
C   1.072   5.336   26.354
C   0.785   4.094   25.729
C   0.482   4.328   24.391
C   0.732   5.796   24.161
C   0.951   2.722   26.387
C   0.181   3.331   23.304
O   0.078   3.598   22.138
C   -0.002   1.911   23.622
N   1.551   8.789   23.594
C   0.944   7.921   22.757
C   0.912   8.523   21.385
C   0.910   10.018   21.665
C   1.475   9.995   23.048
C   2.108   8.070   20.410
C   -0.449   10.792   21.639
C   -0.529   11.886   20.537
N   1.841   10.349   26.023
C   2.044   11.344   25.142
C   2.398   12.590   25.812
C   2.493   12.187   27.165
C   2.168   10.828   27.245
C   2.652   13.889   25.153
C   2.793   12.654   28.503
O   3.179   13.754   28.827
C   2.496   11.488   29.464
C   3.739   11.295   30.172
O   4.842   10.907   29.736
O   3.645   11.802   31.448
C   4.784   11.391   32.461
C   3.135   6.088   34.029
C   2.268   5.482   35.090
C   2.607   5.069   36.289
C   4.057   5.333   36.805
C   1.693   4.426   37.313
C   2.066   3.116   37.977
C   1.433   1.875   37.231
C   0.426   1.148   38.152
C   -1.017   1.113   37.626
C   0.898   -0.284   38.527
C   2.139   -0.325   39.439
C   3.441   -0.852   38.763
C   3.968   -2.221   39.280
C   2.975   -3.281   39.034
C   4.562   -2.231   40.781
C   6.102   -2.623   40.754
C   6.901   -1.289   40.846
C   8.497   -1.432   40.638
C   9.241   -1.322   42.028
C   9.101   -0.489   39.641
H   1.338   4.709   28.308
H   0.315   6.021   22.122
H   1.959   12.057   23.143
H   3.035   8.760   30.441
H   1.410   6.382   30.406
H   4.183   7.039   30.495
H   3.886   5.777   29.153
H   3.491   5.444   30.794
H   0.056   8.222   31.035
H   0.506   9.780   30.668
H   0.549   9.132   33.305
H   1.724   10.171   32.578
H   0.037   2.128   26.408
H   1.343   2.942   27.380
H   1.760   2.222   25.854
H   0.979   1.460   23.772
H   -0.441   1.453   22.735
H   -0.813   1.759   24.334
H   -0.078   8.243   21.023
H   1.557   10.539   20.959
H   2.967   7.747   20.998
H   2.493   8.782   19.680
H   1.838   7.166   19.865
H   -0.607   11.215   22.631
H   -1.282   10.115   21.448
H   -1.544   11.885   20.137
H   0.049   11.641   19.646
H   -0.127   12.745   21.072
H   3.491   13.694   24.485
H   3.045   14.672   25.801
H   1.816   14.157   24.508
H   1.710   11.723   30.182
H   5.449   10.642   32.033
H   4.236   11.000   33.319
H   5.426   12.211   32.782
H   4.065   6.512   34.409
H   3.395   5.292   33.332
H   1.242   5.231   34.821
H   4.828   4.699   36.366
H   4.154   5.253   37.888
H   4.278   6.353   36.492
H   0.669   4.405   36.940
H   1.833   5.071   38.180
H   1.861   3.149   39.047
H   3.150   3.038   37.887
H   2.285   1.307   36.857
H   0.938   2.076   36.281
H   0.336   1.783   39.033
H   -1.134   0.881   36.568
H   -1.533   2.072   37.667
H   -1.681   0.364   38.059
H   1.227   -0.707   37.578
H   0.117   -0.938   38.914
H   2.036   -0.815   40.407
H   2.406   0.730   39.494
H   4.306   -0.195   38.852
H   3.180   -0.880   37.705
H   4.813   -2.358   38.606
H   1.940   -3.014   39.250
H   3.144   -4.240   39.523
H   3.025   -3.492   37.966
H   4.191   -3.055   41.390
H   4.356   -1.285   41.282
H   6.336   -3.150   39.830
H   6.265   -3.174   41.680
H   6.684   -0.839   41.815
H   6.459   -0.723   40.027
H   8.694   -2.420   40.221
H   8.987   -0.382   42.518
H   10.316   -1.472   41.923
H   8.873   -2.135   42.654
H   8.256   0.116   39.311
H   9.454   -0.974   38.731
H   9.862   0.188   40.029

