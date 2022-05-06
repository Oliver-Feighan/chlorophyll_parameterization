%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1351_chromophore_7 TDDFT with wB97XD functional

0 1
Mg   25.520   -0.458   29.661
C   27.473   -1.053   32.623
C   22.742   -0.419   31.616
C   23.845   -0.557   26.824
C   28.574   -0.962   27.795
N   25.229   -0.719   31.923
C   26.112   -0.833   32.927
C   25.455   -0.620   34.274
C   23.920   -0.602   33.864
C   23.940   -0.489   32.347
C   23.009   -1.747   34.461
C   25.971   0.726   34.948
C   26.542   0.650   36.373
C   25.766   -0.188   37.431
O   25.824   -1.403   37.536
O   24.893   0.550   38.229
N   23.520   -0.466   29.299
C   22.615   -0.433   30.210
C   21.297   -0.336   29.467
C   21.608   -0.564   28.079
C   23.043   -0.531   27.986
C   19.953   -0.149   30.171
C   20.650   -0.650   26.957
O   21.004   -0.792   25.791
C   19.128   -0.697   27.164
N   26.176   -0.858   27.536
C   25.254   -0.637   26.593
C   25.901   -0.735   25.156
C   27.461   -0.852   25.453
C   27.434   -0.789   26.995
C   25.240   -1.932   24.334
C   28.316   0.334   24.864
C   29.183   -0.135   23.666
N   27.545   -0.887   30.028
C   28.678   -1.041   29.226
C   29.886   -1.257   29.971
C   29.388   -1.197   31.243
C   28.002   -1.055   31.257
C   31.263   -1.566   29.430
C   29.860   -1.358   32.614
O   31.027   -1.570   33.013
C   28.628   -1.174   33.561
C   28.578   -2.365   34.508
O   28.201   -3.509   34.255
O   28.957   -1.985   35.761
C   29.041   -3.034   36.751
C   24.054   -0.275   39.128
C   23.392   0.773   40.066
C   23.173   0.459   41.330
C   23.052   -0.883   41.966
C   22.612   1.633   42.177
C   23.273   1.956   43.492
C   22.420   2.586   44.612
C   22.506   1.696   45.935
C   21.615   0.435   45.856
C   22.196   2.627   47.149
C   23.351   3.201   47.925
C   23.030   4.679   48.253
C   23.543   5.646   47.212
C   24.279   6.891   47.841
C   22.386   6.222   46.307
C   22.929   6.567   44.899
C   22.950   8.072   44.402
C   21.600   8.627   43.872
C   21.863   9.826   42.828
C   20.800   9.141   45.072
H   21.842   -0.312   32.226
H   23.494   -0.454   25.795
H   29.478   -1.065   27.191
H   25.545   -1.487   34.928
H   23.541   0.353   34.230
H   22.625   -2.202   33.548
H   22.157   -1.333   34.999
H   23.663   -2.402   35.037
H   25.189   1.481   35.024
H   26.731   1.026   34.226
H   26.639   1.656   36.782
H   27.573   0.296   36.348
H   19.904   -0.437   31.221
H   19.220   -0.737   29.619
H   19.574   0.872   30.131
H   18.573   -0.446   26.260
H   19.015   0.156   27.833
H   18.734   -1.663   27.479
H   25.707   0.230   24.687
H   27.807   -1.768   24.974
H   24.867   -2.628   25.086
H   25.871   -2.538   23.684
H   24.379   -1.560   23.779
H   29.053   0.525   25.643
H   27.792   1.271   24.675
H   30.201   -0.291   24.022
H   29.315   0.665   22.937
H   28.760   -1.021   23.192
H   31.219   -2.009   28.435
H   31.632   -2.311   30.136
H   31.844   -0.645   29.384
H   28.743   -0.303   34.206
H   29.103   -4.029   36.309
H   28.172   -2.878   37.390
H   29.966   -2.876   37.305
H   24.659   -1.006   39.664
H   23.333   -0.818   38.516
H   23.474   1.829   39.811
H   24.036   -1.336   42.093
H   22.403   -1.451   41.301
H   22.588   -0.793   42.948
H   21.705   1.148   42.538
H   22.299   2.559   41.695
H   24.077   2.619   43.172
H   23.679   1.055   43.951
H   21.404   2.655   44.225
H   22.790   3.545   44.976
H   23.515   1.306   46.069
H   20.743   0.755   45.285
H   21.269   0.140   46.846
H   22.303   -0.247   45.355
H   21.668   1.938   47.808
H   21.475   3.412   46.923
H   24.327   3.124   47.445
H   23.375   2.804   48.940
H   23.416   4.951   49.235
H   21.960   4.876   48.319
H   24.327   5.162   46.630
H   24.024   7.729   47.193
H   25.363   6.903   47.725
H   23.999   7.223   48.841
H   22.054   7.194   46.670
H   21.461   5.645   46.334
H   22.343   6.066   44.128
H   23.915   6.126   44.752
H   23.589   8.182   43.525
H   23.309   8.786   45.143
H   21.019   7.797   43.469
H   22.890   10.192   42.817
H   21.103   10.595   42.961
H   21.633   9.367   41.867
H   20.064   9.892   44.784
H   21.423   9.541   45.871
H   20.372   8.279   45.585

