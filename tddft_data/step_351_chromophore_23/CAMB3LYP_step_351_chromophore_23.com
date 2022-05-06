%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_351_chromophore_23 TDDFT with cam-b3lyp functional

0 1
Mg   -9.669   41.374   42.997
C   -8.566   38.079   41.747
C   -7.489   42.779   40.966
C   -11.274   44.335   43.646
C   -12.583   39.676   44.050
N   -8.245   40.398   41.516
C   -7.881   39.151   41.211
C   -6.630   39.031   40.394
C   -6.204   40.537   40.246
C   -7.361   41.300   40.997
C   -4.745   40.912   40.564
C   -6.864   38.264   39.028
C   -5.951   37.036   38.797
C   -4.993   37.077   37.655
O   -3.819   37.562   37.678
O   -5.643   36.644   36.542
N   -9.290   43.319   42.577
C   -8.364   43.680   41.660
C   -8.393   45.096   41.506
C   -9.403   45.628   42.341
C   -10.001   44.442   42.955
C   -7.562   45.934   40.514
C   -9.863   47.109   42.478
O   -9.245   48.020   41.962
C   -11.017   47.537   43.390
N   -11.812   41.886   43.490
C   -12.108   43.212   43.809
C   -13.502   43.299   44.451
C   -13.814   41.805   44.761
C   -12.707   41.041   43.957
C   -13.641   44.203   45.671
C   -15.246   41.317   44.324
C   -16.112   40.714   45.451
N   -10.444   39.283   43.069
C   -11.623   38.768   43.530
C   -11.567   37.341   43.584
C   -10.372   37.044   42.898
C   -9.745   38.267   42.570
C   -12.637   36.453   44.069
C   -9.605   35.904   42.463
O   -9.662   34.711   42.653
C   -8.396   36.521   41.601
C   -7.210   36.015   42.190
O   -6.799   36.289   43.340
O   -6.738   34.962   41.468
C   -5.756   34.067   42.133
C   -4.928   36.906   35.269
C   -5.653   36.415   34.044
C   -6.280   37.229   33.194
C   -6.797   38.584   33.475
C   -6.827   36.556   31.956
C   -6.178   36.927   30.573
C   -7.141   37.369   29.478
C   -6.347   37.968   28.232
C   -6.061   36.874   27.176
C   -7.107   39.222   27.791
C   -6.497   40.439   28.494
C   -7.513   41.464   28.963
C   -7.192   42.272   30.240
C   -7.871   41.818   31.505
C   -7.293   43.715   29.958
C   -6.251   44.354   28.998
C   -5.620   45.739   29.407
C   -4.228   46.109   28.912
C   -3.521   47.131   29.809
C   -4.275   46.599   27.500
H   -6.675   43.161   40.347
H   -11.884   45.232   43.777
H   -13.460   39.410   44.645
H   -5.832   38.570   40.975
H   -6.266   40.843   39.202
H   -4.408   41.751   39.954
H   -4.065   40.085   40.363
H   -4.770   41.052   41.645
H   -6.542   38.945   38.240
H   -7.920   38.010   38.932
H   -6.633   36.186   38.797
H   -5.302   36.926   39.666
H   -8.199   46.690   40.056
H   -7.127   45.289   39.751
H   -6.767   46.388   41.106
H   -11.030   48.609   43.592
H   -10.864   47.095   44.375
H   -12.001   47.347   42.961
H   -14.277   43.617   43.754
H   -13.539   41.682   45.809
H   -12.693   44.714   45.841
H   -14.098   43.580   46.440
H   -14.317   44.999   45.357
H   -15.052   40.597   43.529
H   -15.887   42.073   43.870
H   -16.386   39.699   45.162
H   -17.030   41.298   45.510
H   -15.589   40.696   46.407
H   -13.138   36.908   44.923
H   -12.196   35.501   44.364
H   -13.360   36.316   43.264
H   -8.458   36.237   40.551
H   -6.121   33.356   42.874
H   -4.914   34.599   42.578
H   -5.333   33.428   41.358
H   -4.022   36.317   35.125
H   -4.638   37.926   35.017
H   -5.393   35.425   33.670
H   -6.917   38.727   34.549
H   -7.705   38.831   32.926
H   -5.998   39.250   33.148
H   -7.875   36.837   31.849
H   -6.843   35.473   32.086
H   -5.662   36.092   30.099
H   -5.421   37.691   30.755
H   -7.661   38.151   30.031
H   -7.856   36.577   29.254
H   -5.387   38.288   28.639
H   -5.060   36.444   27.215
H   -6.221   37.253   26.167
H   -6.761   36.041   27.122
H   -8.189   39.158   27.909
H   -6.890   39.358   26.732
H   -5.786   40.970   27.862
H   -6.069   39.995   29.393
H   -8.490   41.010   29.128
H   -7.703   42.148   28.136
H   -6.121   42.107   30.353
H   -8.611   41.036   31.337
H   -8.577   42.560   31.878
H   -7.232   41.583   32.357
H   -7.343   44.226   30.920
H   -8.319   43.821   29.606
H   -6.839   44.558   28.103
H   -5.541   43.560   28.764
H   -5.553   45.797   30.494
H   -6.350   46.385   28.919
H   -3.613   45.212   28.987
H   -4.145   47.681   30.514
H   -3.182   48.035   29.304
H   -2.593   46.803   30.278
H   -4.709   47.599   27.512
H   -5.019   46.011   26.963
H   -3.330   46.726   26.971

