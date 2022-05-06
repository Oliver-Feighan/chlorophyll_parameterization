%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1051_chromophore_19 TDDFT with PBE1PBE functional

0 1
Mg   25.711   50.525   26.587
C   23.643   51.278   29.354
C   28.097   49.561   28.715
C   27.628   50.284   23.899
C   23.145   52.016   24.592
N   25.754   50.331   28.826
C   24.883   50.706   29.756
C   25.430   50.525   31.099
C   26.886   49.956   30.924
C   26.955   49.957   29.394
C   28.160   50.712   31.644
C   24.458   49.582   31.936
C   23.765   49.962   33.370
C   24.669   50.574   34.472
O   25.343   51.607   34.383
O   24.692   49.823   35.651
N   27.522   49.824   26.348
C   28.353   49.435   27.350
C   29.488   48.875   26.810
C   29.429   49.101   25.419
C   28.191   49.723   25.142
C   30.403   48.005   27.735
C   30.538   48.855   24.406
O   30.279   49.099   23.200
C   31.858   48.267   24.766
N   25.532   51.334   24.571
C   26.459   51.003   23.670
C   26.115   51.615   22.274
C   24.529   51.777   22.415
C   24.420   51.780   23.984
C   26.792   53.010   21.877
C   23.730   50.562   21.913
C   22.376   50.818   21.096
N   23.796   51.313   26.834
C   22.885   51.753   25.927
C   21.613   52.001   26.545
C   21.930   51.895   27.907
C   23.218   51.459   28.008
C   20.275   52.258   25.756
C   21.317   52.130   29.241
O   20.242   52.546   29.610
C   22.520   51.744   30.256
C   22.843   52.979   30.986
O   23.327   53.905   30.486
O   22.434   52.787   32.291
C   22.446   53.990   33.064
C   25.520   50.385   36.714
C   25.262   49.758   38.091
C   25.848   49.761   39.306
C   27.176   50.577   39.484
C   25.334   48.883   40.411
C   25.666   47.410   40.268
C   26.964   46.953   41.024
C   26.916   45.504   41.535
C   26.627   44.388   40.449
C   26.092   45.315   42.854
C   26.994   45.396   44.066
C   26.296   46.001   45.350
C   27.016   47.288   45.951
C   26.860   47.179   47.476
C   26.534   48.630   45.413
C   27.674   49.617   45.111
C   27.308   51.016   45.621
C   26.856   51.925   44.377
C   28.075   52.639   43.709
C   25.864   52.962   44.957
H   28.863   49.183   29.395
H   28.289   50.190   23.035
H   22.251   52.377   24.080
H   25.434   51.457   31.665
H   26.932   48.878   31.077
H   28.767   51.307   30.961
H   28.858   49.948   31.987
H   27.739   51.334   32.434
H   25.049   48.710   32.216
H   23.668   49.225   31.274
H   23.382   49.025   33.774
H   22.864   50.547   33.186
H   30.040   47.731   28.725
H   31.392   48.429   27.907
H   30.595   47.069   27.211
H   32.431   48.381   23.846
H   31.739   47.257   25.159
H   32.342   48.839   25.558
H   26.342   50.956   21.436
H   24.216   52.738   22.005
H   27.804   53.215   22.226
H   26.223   53.917   22.080
H   26.929   52.943   20.797
H   23.568   49.943   22.795
H   24.330   49.909   21.279
H   21.529   50.194   21.383
H   22.565   50.756   20.025
H   22.027   51.833   21.282
H   20.390   53.301   25.461
H   19.420   52.042   26.397
H   20.188   51.712   24.817
H   22.136   50.975   30.926
H   21.403   54.257   33.234
H   22.911   54.859   32.597
H   22.802   53.742   34.064
H   25.249   51.416   36.941
H   26.602   50.333   36.597
H   24.434   49.066   37.939
H   26.743   51.489   39.894
H   27.667   50.728   38.523
H   27.852   50.067   40.171
H   24.259   48.916   40.591
H   25.769   49.247   41.341
H   25.792   47.266   39.195
H   24.810   46.866   40.669
H   27.224   47.568   41.885
H   27.603   46.932   40.142
H   27.970   45.263   41.670
H   26.587   44.785   39.435
H   25.787   43.707   40.589
H   27.495   43.731   40.391
H   25.549   44.370   42.885
H   25.338   46.102   42.824
H   27.914   45.963   43.923
H   27.317   44.393   44.346
H   26.382   45.228   46.113
H   25.269   46.317   45.170
H   28.062   47.165   45.669
H   27.831   46.923   47.901
H   26.107   46.448   47.768
H   26.566   48.159   47.852
H   25.728   49.164   45.918
H   25.982   48.427   44.496
H   27.889   49.506   44.048
H   28.567   49.268   45.628
H   28.237   51.377   46.063
H   26.509   51.185   46.342
H   26.283   51.263   43.727
H   27.832   53.696   43.600
H   28.210   52.267   42.694
H   29.047   52.390   44.133
H   24.832   52.628   45.060
H   25.765   53.871   44.364
H   26.171   53.304   45.945

