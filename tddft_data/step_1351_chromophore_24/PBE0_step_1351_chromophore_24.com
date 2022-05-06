%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1351_chromophore_24 TDDFT with PBE1PBE functional

0 1
Mg   -0.435   43.653   24.990
C   2.054   43.324   27.403
C   -2.779   42.952   27.554
C   -2.928   43.523   22.605
C   1.940   43.888   22.615
N   -0.388   43.368   27.247
C   0.765   43.239   28.004
C   0.359   43.093   29.442
C   -1.124   42.697   29.453
C   -1.472   42.967   28.032
C   -1.447   41.237   29.867
C   0.574   44.418   30.200
C   0.458   44.247   31.715
C   0.365   42.970   32.354
O   1.290   42.164   32.463
O   -0.846   42.897   33.062
N   -2.543   43.350   25.090
C   -3.305   43.117   26.226
C   -4.700   43.115   25.828
C   -4.771   43.119   24.390
C   -3.378   43.367   23.965
C   -5.833   42.956   26.777
C   -5.967   43.093   23.368
O   -5.885   43.055   22.162
C   -7.275   42.838   24.052
N   -0.396   43.829   22.870
C   -1.545   43.703   22.138
C   -1.149   43.607   20.655
C   0.350   44.061   20.673
C   0.682   43.885   22.152
C   -1.273   42.128   20.098
C   0.735   45.479   20.103
C   -0.084   46.631   20.666
N   1.527   43.604   24.980
C   2.406   43.820   23.934
C   3.764   43.667   24.396
C   3.709   43.521   25.739
C   2.350   43.543   26.052
C   5.000   43.558   23.586
C   4.478   43.313   26.996
O   5.667   43.101   27.234
C   3.357   43.288   28.129
C   3.553   41.966   28.791
O   3.011   40.918   28.481
O   4.298   42.251   29.936
C   4.300   41.218   30.933
C   -0.841   41.753   34.025
C   -2.311   41.776   34.527
C   -2.847   42.172   35.747
C   -2.030   42.865   36.748
C   -4.377   42.115   36.014
C   -5.145   43.384   35.417
C   -6.726   43.173   35.069
C   -7.597   43.583   36.282
C   -8.122   44.981   36.101
C   -8.785   42.629   36.504
C   -8.427   41.511   37.510
C   -8.920   41.930   38.913
C   -10.450   41.646   39.169
C   -11.288   42.889   39.426
C   -10.648   40.557   40.233
C   -11.015   39.187   39.660
C   -12.348   38.607   40.156
C   -13.119   37.810   39.026
C   -14.273   38.613   38.342
C   -13.782   36.600   39.687
H   -3.540   42.712   28.300
H   -3.629   43.474   21.769
H   2.735   43.935   21.867
H   1.068   42.331   29.769
H   -1.684   43.346   30.125
H   -0.543   40.709   30.171
H   -1.841   40.643   29.043
H   -2.180   41.325   30.669
H   -0.174   45.136   29.863
H   1.557   44.833   29.979
H   -0.268   44.934   32.148
H   1.399   44.630   32.109
H   -5.668   42.954   27.854
H   -6.255   41.982   26.531
H   -6.553   43.773   26.809
H   -7.539   43.697   24.670
H   -7.299   41.883   24.577
H   -7.960   42.698   23.216
H   -1.716   44.292   20.024
H   1.048   43.402   20.156
H   -2.090   41.730   20.701
H   -0.321   41.621   20.255
H   -1.663   42.178   19.081
H   0.647   45.334   19.026
H   1.730   45.737   20.466
H   -0.328   47.247   19.801
H   0.525   47.181   21.383
H   -1.025   46.394   21.162
H   4.841   43.457   22.513
H   5.499   42.710   24.055
H   5.738   44.346   23.734
H   3.451   44.133   28.811
H   3.338   41.194   31.445
H   5.065   41.405   31.687
H   4.567   40.248   30.514
H   -0.094   41.685   34.816
H   -0.684   40.796   33.527
H   -3.064   41.382   33.845
H   -2.143   42.272   37.656
H   -2.470   43.845   36.938
H   -0.970   42.896   36.498
H   -4.600   42.052   37.079
H   -4.763   41.139   35.721
H   -4.630   43.794   34.549
H   -4.991   44.214   36.106
H   -6.897   42.114   34.878
H   -7.005   43.706   34.160
H   -6.980   43.522   37.179
H   -8.306   45.335   35.087
H   -7.375   45.703   36.432
H   -9.049   45.159   36.646
H   -9.028   42.136   35.563
H   -9.612   43.221   36.898
H   -7.387   41.184   37.483
H   -8.984   40.705   37.033
H   -8.759   42.985   39.139
H   -8.139   41.468   39.517
H   -10.802   41.205   38.237
H   -12.216   42.727   39.974
H   -11.545   43.170   38.404
H   -10.741   43.726   39.860
H   -11.498   40.889   40.829
H   -9.774   40.457   40.876
H   -10.321   38.417   39.997
H   -11.007   39.230   38.571
H   -12.994   39.418   40.492
H   -12.159   37.867   40.933
H   -12.399   37.658   38.221
H   -14.043   38.567   37.277
H   -14.075   39.675   38.485
H   -15.290   38.376   38.653
H   -13.775   36.566   40.777
H   -13.138   35.784   39.359
H   -14.805   36.365   39.394

