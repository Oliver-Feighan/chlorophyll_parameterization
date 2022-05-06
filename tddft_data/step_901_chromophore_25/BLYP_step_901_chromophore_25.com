%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_901_chromophore_25 TDDFT with blyp functional

0 1
Mg   -2.395   34.821   26.456
C   -3.411   32.898   29.356
C   -1.266   37.298   28.547
C   -2.369   36.951   23.830
C   -4.275   32.626   24.464
N   -2.417   35.112   28.709
C   -2.791   34.195   29.668
C   -2.368   34.720   31.147
C   -1.719   36.076   30.787
C   -1.719   36.118   29.252
C   -2.296   37.341   31.474
C   -1.444   33.732   32.024
C   -1.978   33.326   33.444
C   -0.979   33.292   34.519
O   -0.089   32.521   34.669
O   -1.244   34.223   35.481
N   -1.753   36.804   26.237
C   -1.351   37.647   27.201
C   -0.763   38.902   26.563
C   -1.202   38.711   25.145
C   -1.879   37.435   24.992
C   0.019   39.993   27.324
C   -1.060   39.744   24.076
O   -1.602   39.755   23.012
C   -0.134   40.876   24.253
N   -3.322   34.901   24.377
C   -3.012   35.813   23.525
C   -3.400   35.384   22.109
C   -3.814   33.839   22.334
C   -3.827   33.750   23.832
C   -4.400   36.360   21.472
C   -2.985   32.869   21.415
C   -1.581   32.495   22.044
N   -3.559   33.053   26.751
C   -4.181   32.233   25.841
C   -4.686   31.073   26.574
C   -4.414   31.327   27.914
C   -3.779   32.567   27.982
C   -5.576   30.037   26.099
C   -4.596   30.877   29.259
O   -5.141   29.873   29.721
C   -3.978   31.896   30.278
C   -5.062   32.349   31.277
O   -6.119   32.965   30.977
O   -4.811   31.886   32.511
C   -5.842   31.977   33.541
C   -0.454   34.195   36.738
C   -0.884   35.349   37.683
C   -1.708   35.315   38.707
C   -2.326   34.008   39.253
C   -2.222   36.563   39.418
C   -1.329   37.201   40.471
C   -2.199   38.214   41.293
C   -2.430   37.796   42.799
C   -3.789   38.167   43.344
C   -1.261   38.337   43.743
C   -0.662   37.256   44.668
C   0.874   37.238   44.880
C   1.417   35.825   44.985
C   0.861   34.897   46.068
C   3.033   35.789   45.047
C   3.569   34.965   43.834
C   4.404   35.813   42.828
C   3.968   35.547   41.319
C   2.475   36.055   41.010
C   4.236   33.998   40.953
H   -0.778   38.081   29.132
H   -2.370   37.534   22.907
H   -4.835   31.905   23.865
H   -3.220   34.882   31.807
H   -0.681   36.170   31.108
H   -2.846   37.026   32.361
H   -3.099   37.906   31.001
H   -1.552   38.074   31.783
H   -0.446   34.157   32.133
H   -1.173   32.830   31.475
H   -2.319   32.292   33.480
H   -2.890   33.890   33.639
H   -0.606   40.882   27.408
H   0.981   40.314   26.923
H   0.265   39.685   28.340
H   -0.566   41.547   24.995
H   0.078   41.452   23.352
H   0.850   40.496   24.527
H   -2.532   35.378   21.450
H   -4.859   33.733   22.042
H   -5.099   35.789   20.861
H   -3.888   37.002   20.755
H   -4.981   36.838   22.260
H   -2.803   33.439   20.504
H   -3.513   31.927   21.264
H   -0.872   33.186   21.586
H   -1.335   31.499   21.677
H   -1.490   32.698   23.111
H   -5.304   29.828   25.064
H   -6.604   30.393   26.168
H   -5.542   29.069   26.598
H   -3.043   31.507   30.680
H   -5.631   31.245   34.320
H   -6.834   31.788   33.132
H   -5.810   32.940   34.049
H   0.629   34.294   36.672
H   -0.687   33.229   37.186
H   -0.568   36.360   37.428
H   -1.608   33.188   39.248
H   -3.151   33.668   38.627
H   -2.777   34.161   40.234
H   -3.156   36.409   39.959
H   -2.480   37.324   38.682
H   -0.512   37.745   39.999
H   -0.921   36.503   41.203
H   -3.170   38.432   40.848
H   -1.551   39.088   41.232
H   -2.360   36.709   42.834
H   -4.261   37.462   44.029
H   -4.479   38.131   42.501
H   -3.734   39.159   43.792
H   -1.516   39.101   44.477
H   -0.539   38.766   43.049
H   -0.957   36.249   44.371
H   -1.036   37.338   45.688
H   0.994   37.759   45.830
H   1.334   37.764   44.044
H   1.084   35.396   44.040
H   -0.134   34.502   45.862
H   0.770   35.441   47.008
H   1.599   34.117   46.253
H   3.374   35.305   45.963
H   3.439   36.800   45.074
H   2.798   34.338   43.388
H   4.188   34.199   44.301
H   5.480   35.753   42.988
H   4.160   36.872   42.919
H   4.628   36.097   40.649
H   2.079   36.549   41.897
H   1.819   35.212   40.792
H   2.498   36.753   40.172
H   4.942   33.950   40.124
H   3.329   33.482   40.638
H   4.525   33.340   41.772

