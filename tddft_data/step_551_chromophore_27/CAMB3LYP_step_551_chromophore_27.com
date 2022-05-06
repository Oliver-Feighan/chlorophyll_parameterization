%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_551_chromophore_27 TDDFT with cam-b3lyp functional

0 1
Mg   -5.018   24.272   27.744
C   -3.221   26.224   29.960
C   -5.889   22.232   30.465
C   -6.126   21.952   25.661
C   -3.573   26.025   25.014
N   -4.578   24.206   29.964
C   -3.899   25.240   30.633
C   -4.093   25.108   32.144
C   -4.609   23.609   32.234
C   -5.139   23.370   30.809
C   -3.535   22.550   32.652
C   -5.072   26.184   32.767
C   -4.661   26.850   34.024
C   -5.628   26.729   35.157
O   -6.209   27.727   35.647
O   -5.420   25.498   35.761
N   -6.004   22.363   28.050
C   -6.328   21.735   29.228
C   -6.922   20.400   28.923
C   -7.064   20.384   27.510
C   -6.434   21.647   27.009
C   -7.174   19.422   30.014
C   -7.516   19.177   26.677
O   -7.503   19.236   25.463
C   -8.032   17.823   27.293
N   -4.955   24.083   25.736
C   -5.599   23.073   25.077
C   -5.919   23.440   23.570
C   -4.921   24.721   23.394
C   -4.420   24.943   24.821
C   -5.799   22.296   22.401
C   -5.538   26.017   22.772
C   -4.774   26.689   21.647
N   -3.655   25.893   27.431
C   -3.112   26.488   26.253
C   -2.194   27.523   26.637
C   -2.212   27.489   28.066
C   -3.060   26.414   28.499
C   -1.379   28.312   25.698
C   -1.694   28.066   29.363
O   -0.823   28.930   29.537
C   -2.363   27.305   30.577
C   -1.219   26.854   31.417
O   -0.502   25.934   31.133
O   -1.150   27.633   32.535
C   -0.129   27.179   33.494
C   -5.968   25.104   37.085
C   -6.735   23.833   36.882
C   -7.181   22.945   37.816
C   -6.838   23.116   39.269
C   -7.876   21.644   37.371
C   -9.343   21.714   37.408
C   -10.016   20.763   38.434
C   -11.390   20.202   37.965
C   -12.477   21.250   38.316
C   -11.602   18.879   38.642
C   -10.770   17.680   38.103
C   -10.058   16.797   39.224
C   -8.573   16.321   38.934
C   -7.542   17.207   39.694
C   -8.238   14.854   39.248
C   -7.243   14.259   38.230
C   -5.937   13.664   38.877
C   -4.571   14.349   38.691
C   -3.914   13.918   37.336
C   -3.648   14.171   39.847
H   -6.182   21.625   31.324
H   -6.501   21.279   24.887
H   -3.201   26.591   24.158
H   -3.157   25.132   32.703
H   -5.401   23.583   32.982
H   -2.599   23.043   32.915
H   -3.293   21.840   31.862
H   -3.958   22.114   33.557
H   -6.030   25.665   32.810
H   -5.272   26.944   32.011
H   -4.620   27.919   33.814
H   -3.645   26.605   34.334
H   -7.234   19.843   31.018
H   -6.402   18.655   29.963
H   -8.201   19.115   29.815
H   -7.612   17.468   28.234
H   -7.939   16.956   26.639
H   -9.071   18.071   27.511
H   -6.958   23.744   23.440
H   -4.029   24.412   22.848
H   -5.267   22.623   21.507
H   -6.824   22.038   22.135
H   -5.287   21.413   22.783
H   -5.702   26.847   23.459
H   -6.476   25.580   22.432
H   -5.430   27.239   20.974
H   -3.997   26.066   21.203
H   -4.175   27.489   22.081
H   -1.003   29.230   26.152
H   -2.008   28.574   24.848
H   -0.585   27.760   25.196
H   -3.034   27.913   31.184
H   -0.473   27.123   34.527
H   0.718   27.865   33.510
H   0.232   26.171   33.289
H   -6.467   25.978   37.503
H   -5.044   24.842   37.599
H   -6.796   23.455   35.862
H   -7.722   23.611   39.672
H   -6.012   23.780   39.523
H   -6.596   22.198   39.804
H   -7.542   20.795   37.968
H   -7.641   21.658   36.307
H   -9.818   21.748   36.427
H   -9.460   22.707   37.843
H   -10.089   21.235   39.414
H   -9.443   19.851   38.602
H   -11.392   20.039   36.887
H   -13.201   20.903   39.054
H   -12.948   21.419   37.348
H   -12.075   22.151   38.780
H   -12.623   18.669   38.325
H   -11.594   19.023   39.723
H   -10.147   17.994   37.266
H   -11.494   16.997   37.658
H   -10.653   15.895   39.365
H   -10.076   17.364   40.155
H   -8.371   16.505   37.879
H   -7.027   16.672   40.492
H   -7.906   18.118   40.169
H   -6.710   17.489   39.050
H   -9.092   14.189   39.380
H   -7.957   14.825   40.301
H   -6.919   15.012   37.511
H   -7.710   13.454   37.661
H   -5.722   12.697   38.421
H   -6.291   13.406   39.875
H   -4.830   15.397   38.547
H   -4.269   14.632   36.592
H   -4.208   12.953   36.924
H   -2.826   13.983   37.351
H   -4.056   13.363   40.455
H   -3.639   15.134   40.359
H   -2.663   13.911   39.459
