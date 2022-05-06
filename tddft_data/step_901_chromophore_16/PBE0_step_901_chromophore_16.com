%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_901_chromophore_16 TDDFT with PBE1PBE functional

0 1
Mg   40.478   41.694   27.078
C   39.677   44.091   29.616
C   41.315   39.487   29.601
C   41.407   39.643   24.644
C   40.106   44.439   24.854
N   40.475   41.792   29.376
C   40.063   42.862   30.162
C   40.031   42.429   31.664
C   40.613   40.945   31.632
C   40.829   40.684   30.132
C   41.858   40.644   32.530
C   38.604   42.615   32.328
C   38.331   41.741   33.552
C   37.611   42.464   34.590
O   36.429   42.816   34.685
O   38.443   42.588   35.594
N   41.270   39.795   27.103
C   41.510   39.033   28.240
C   42.043   37.724   27.779
C   42.094   37.753   26.345
C   41.596   39.083   25.952
C   42.360   36.570   28.820
C   42.547   36.716   25.346
O   42.810   36.954   24.162
C   42.882   35.333   25.890
N   40.770   42.012   25.028
C   41.143   40.990   24.223
C   41.291   41.446   22.790
C   40.802   42.889   22.852
C   40.478   43.171   24.339
C   42.761   41.317   22.211
C   39.718   43.141   21.793
C   38.315   43.369   22.369
N   40.048   43.803   27.162
C   39.962   44.740   26.167
C   39.501   46.021   26.691
C   39.369   45.821   28.062
C   39.800   44.462   28.265
C   39.148   47.342   26.062
C   38.959   46.396   29.332
O   38.608   47.500   29.528
C   39.171   45.325   30.407
C   40.094   45.871   31.469
O   41.312   45.894   31.357
O   39.379   46.338   32.537
C   40.183   46.855   33.629
C   37.983   43.465   36.705
C   38.553   42.695   37.933
C   38.331   42.975   39.250
C   37.509   44.213   39.808
C   38.979   42.052   40.279
C   38.041   41.240   41.313
C   38.166   39.691   41.080
C   38.419   38.833   42.388
C   39.636   37.968   42.279
C   37.144   38.048   42.792
C   36.450   38.684   44.018
C   37.021   38.209   45.396
C   35.985   37.323   46.181
C   36.306   37.221   47.654
C   35.908   35.850   45.512
C   34.632   35.716   44.614
C   33.548   34.724   45.220
C   32.593   35.617   45.992
C   31.227   35.609   45.333
C   32.556   35.277   47.478
H   41.410   38.813   30.455
H   41.533   38.942   23.816
H   39.826   45.144   24.069
H   40.676   43.098   32.235
H   39.852   40.222   31.928
H   41.730   39.688   33.039
H   42.002   41.426   33.275
H   42.715   40.492   31.873
H   37.768   42.363   31.676
H   38.644   43.661   32.632
H   39.226   41.350   34.037
H   37.713   40.866   33.347
H   41.630   35.761   28.797
H   42.236   36.804   29.877
H   43.370   36.203   28.637
H   41.960   35.094   26.420
H   43.799   35.342   26.480
H   43.032   34.660   25.045
H   40.694   40.898   22.060
H   41.602   43.593   22.625
H   43.216   40.527   22.809
H   43.363   42.213   22.359
H   42.771   40.979   21.174
H   39.787   42.367   21.029
H   40.071   44.049   21.303
H   37.862   44.327   22.113
H   38.168   43.152   23.427
H   37.765   42.626   21.790
H   38.949   48.081   26.838
H   38.279   47.231   25.413
H   39.914   47.822   25.454
H   38.217   45.120   30.893
H   40.902   47.597   33.280
H   40.792   46.131   34.169
H   39.561   47.410   34.331
H   36.917   43.623   36.865
H   38.614   44.353   36.741
H   39.144   41.791   37.787
H   38.054   44.874   40.482
H   36.752   43.832   40.494
H   37.022   44.799   39.029
H   39.625   42.590   40.973
H   39.428   41.245   39.700
H   37.037   41.611   41.106
H   38.487   41.587   42.244
H   38.845   39.381   40.285
H   37.189   39.348   40.740
H   38.510   39.527   43.224
H   39.497   37.019   42.797
H   40.493   38.457   42.742
H   39.884   37.688   41.255
H   37.324   36.976   42.866
H   36.303   38.125   42.102
H   35.405   38.463   43.801
H   36.635   39.753   44.126
H   37.385   39.085   45.933
H   37.891   37.584   45.193
H   35.035   37.835   46.028
H   35.413   36.820   48.134
H   36.599   38.158   48.126
H   37.070   36.444   47.627
H   35.869   35.154   46.350
H   36.777   35.557   44.923
H   34.992   35.390   43.639
H   34.184   36.708   44.551
H   34.110   34.103   45.919
H   33.005   34.017   44.594
H   32.828   36.680   45.941
H   30.589   36.419   45.688
H   30.649   34.688   45.416
H   31.410   35.873   44.291
H   31.710   34.629   47.704
H   32.504   36.230   48.005
H   33.470   34.770   47.788

