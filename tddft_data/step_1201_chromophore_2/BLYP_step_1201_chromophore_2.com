%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1201_chromophore_2 TDDFT with blyp functional

0 1
Mg   2.993   0.749   44.097
C   5.878   2.556   43.627
C   1.295   3.230   42.350
C   0.482   -1.407   43.619
C   5.057   -1.954   45.130
N   3.587   2.572   43.003
C   4.794   3.207   43.014
C   4.792   4.543   42.263
C   3.281   4.809   42.069
C   2.638   3.467   42.568
C   2.703   6.117   42.756
C   5.703   4.554   40.941
C   6.178   3.260   40.238
C   6.256   3.150   38.650
O   5.848   4.019   37.923
O   6.829   1.944   38.195
N   1.115   0.975   43.358
C   0.599   2.047   42.689
C   -0.780   1.856   42.471
C   -1.101   0.567   42.914
C   0.165   -0.042   43.283
C   -1.565   2.930   41.895
C   -2.475   -0.096   42.978
O   -3.458   0.533   42.621
C   -2.679   -1.538   43.341
N   2.849   -1.383   44.227
C   1.613   -1.980   44.144
C   1.679   -3.467   44.531
C   3.193   -3.636   44.905
C   3.745   -2.244   44.787
C   0.634   -3.839   45.701
C   3.979   -4.738   44.120
C   4.575   -5.775   45.092
N   4.989   0.353   44.445
C   5.708   -0.774   44.872
C   7.117   -0.384   45.033
C   7.198   0.866   44.578
C   5.894   1.299   44.216
C   8.214   -1.248   45.650
C   8.111   1.954   44.323
O   9.380   1.939   44.471
C   7.319   3.141   43.805
C   7.409   4.252   44.866
O   8.398   4.907   45.229
O   6.136   4.604   45.289
C   5.897   5.843   46.122
C   6.997   1.757   36.723
C   5.590   1.473   36.159
C   5.270   0.910   35.006
C   6.258   0.348   33.958
C   3.842   1.038   34.531
C   3.432   2.549   34.185
C   2.783   2.887   32.846
C   1.552   1.975   32.550
C   1.809   1.159   31.272
C   0.235   2.731   32.487
C   0.026   3.695   31.357
C   -1.236   3.474   30.479
C   -1.183   4.185   29.152
C   -2.059   5.469   29.172
C   -1.662   3.299   27.946
C   -0.931   3.685   26.669
C   -1.654   3.078   25.428
C   -2.041   4.161   24.450
C   -3.463   3.968   23.789
C   -1.043   4.280   23.274
H   0.790   4.016   41.785
H   -0.231   -2.232   43.673
H   5.604   -2.844   45.450
H   5.242   5.279   42.929
H   3.121   4.914   40.996
H   3.327   6.936   42.397
H   2.695   6.068   43.845
H   1.692   6.299   42.390
H   6.567   5.184   41.151
H   5.163   5.003   40.108
H   5.688   2.377   40.648
H   7.164   3.001   40.623
H   -2.629   2.902   42.130
H   -1.235   2.903   40.856
H   -1.236   3.881   42.313
H   -2.234   -1.793   44.303
H   -2.187   -2.147   42.582
H   -3.751   -1.732   43.349
H   1.316   -4.054   43.688
H   3.236   -3.794   45.983
H   -0.353   -4.029   45.278
H   0.565   -3.076   46.476
H   0.970   -4.727   46.236
H   4.817   -4.288   43.588
H   3.325   -5.132   43.343
H   4.198   -5.697   46.112
H   5.650   -5.614   45.173
H   4.406   -6.784   44.717
H   8.197   -1.282   46.739
H   9.190   -0.839   45.390
H   8.172   -2.275   45.288
H   7.850   3.464   42.910
H   6.281   6.687   45.550
H   6.383   5.849   47.099
H   4.842   6.081   46.254
H   7.641   0.891   36.565
H   7.522   2.654   36.393
H   4.778   1.723   36.842
H   6.182   0.776   32.959
H   6.061   -0.706   33.765
H   7.262   0.549   34.333
H   3.241   0.702   35.376
H   3.606   0.293   33.771
H   4.307   3.192   34.279
H   2.669   2.829   34.912
H   3.511   2.684   32.060
H   2.605   3.961   32.789
H   1.490   1.219   33.333
H   1.960   1.775   30.386
H   0.915   0.589   31.020
H   2.592   0.426   31.470
H   0.215   3.304   33.414
H   -0.616   2.052   32.440
H   0.857   3.778   30.657
H   0.154   4.718   31.710
H   -2.097   3.725   31.099
H   -1.192   2.409   30.252
H   -0.141   4.391   28.907
H   -1.340   6.248   28.919
H   -2.409   5.756   30.164
H   -2.903   5.433   28.484
H   -2.739   3.428   27.838
H   -1.524   2.232   28.122
H   0.032   3.202   26.505
H   -0.713   4.743   26.523
H   -2.545   2.563   25.787
H   -0.999   2.323   24.992
H   -1.980   5.126   24.955
H   -4.220   4.677   24.124
H   -3.899   3.042   24.164
H   -3.359   3.976   22.704
H   -0.749   5.275   22.942
H   -1.281   3.599   22.457
H   -0.113   3.847   23.641

