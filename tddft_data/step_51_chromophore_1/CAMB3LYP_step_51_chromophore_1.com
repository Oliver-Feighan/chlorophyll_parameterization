%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_51_chromophore_1 TDDFT with cam-b3lyp functional

0 1
Mg   -1.684   17.297   26.952
C   -2.229   14.960   29.610
C   -2.851   19.747   29.107
C   -1.666   19.543   24.459
C   -2.023   14.749   24.675
N   -2.574   17.250   29.091
C   -2.377   16.302   30.048
C   -2.734   16.793   31.509
C   -3.173   18.243   31.173
C   -2.771   18.486   29.701
C   -4.671   18.518   31.540
C   -1.487   16.631   32.430
C   -1.351   17.674   33.652
C   -0.533   17.175   34.880
O   0.674   16.968   34.926
O   -1.297   17.016   35.987
N   -1.906   19.371   26.847
C   -2.439   20.179   27.866
C   -2.199   21.589   27.405
C   -1.763   21.479   26.053
C   -1.742   20.143   25.724
C   -2.453   22.735   28.383
C   -1.467   22.641   25.093
O   -0.989   22.423   23.977
C   -1.687   24.055   25.589
N   -1.920   17.169   24.896
C   -1.848   18.215   24.060
C   -2.073   17.776   22.611
C   -2.260   16.230   22.695
C   -2.074   16.039   24.195
C   -3.257   18.507   21.884
C   -1.530   15.193   21.748
C   -0.168   14.562   22.120
N   -1.847   15.213   26.969
C   -1.886   14.279   25.986
C   -1.930   13.006   26.532
C   -2.108   13.205   27.932
C   -2.002   14.599   28.185
C   -1.989   11.805   25.652
C   -2.272   12.553   29.280
O   -2.418   11.367   29.486
C   -2.424   13.609   30.387
C   -1.451   13.265   31.419
O   -0.220   13.435   31.354
O   -2.063   12.646   32.504
C   -1.232   12.292   33.673
C   -0.743   16.502   37.245
C   -0.741   17.678   38.271
C   -0.480   17.616   39.600
C   -0.072   16.375   40.278
C   -0.788   18.822   40.449
C   0.120   20.076   40.045
C   0.343   21.068   41.185
C   1.724   21.815   41.045
C   1.911   22.323   39.588
C   2.920   21.064   41.694
C   4.115   22.079   41.876
C   4.911   21.759   43.150
C   4.452   22.557   44.404
C   5.326   23.876   44.623
C   4.519   21.512   45.615
C   3.111   20.936   45.849
C   2.489   21.582   47.150
C   0.952   21.995   46.978
C   1.040   23.486   46.631
C   0.137   21.820   48.240
H   -3.043   20.585   29.780
H   -1.767   20.287   23.666
H   -2.281   13.919   24.014
H   -3.561   16.191   31.886
H   -2.671   18.985   31.793
H   -5.141   18.876   30.624
H   -4.668   19.412   32.163
H   -5.116   17.706   32.115
H   -0.549   16.480   31.895
H   -1.740   15.668   32.872
H   -2.369   17.894   33.973
H   -0.920   18.621   33.327
H   -2.577   22.356   29.398
H   -3.348   23.248   28.031
H   -1.551   23.347   28.377
H   -2.650   24.029   26.100
H   -1.633   24.654   24.680
H   -0.843   24.245   26.252
H   -1.148   18.018   22.088
H   -3.312   15.949   22.645
H   -3.967   18.965   22.573
H   -3.844   17.827   21.267
H   -2.805   19.242   21.218
H   -1.508   15.741   20.806
H   -2.198   14.340   21.627
H   -0.306   13.556   22.516
H   0.321   15.208   22.851
H   0.536   14.427   21.299
H   -1.315   11.972   24.811
H   -3.024   11.625   25.364
H   -1.621   10.930   26.188
H   -3.411   13.582   30.847
H   -0.979   13.200   34.219
H   -0.303   11.850   33.312
H   -1.707   11.606   34.375
H   0.289   16.161   37.159
H   -1.245   15.626   37.656
H   -1.147   18.581   37.815
H   -0.928   16.072   40.881
H   0.706   16.633   40.997
H   0.162   15.503   39.667
H   -0.770   18.591   41.514
H   -1.781   19.158   40.152
H   -0.450   20.489   39.212
H   1.032   19.605   39.680
H   0.388   20.694   42.208
H   -0.403   21.857   41.096
H   1.598   22.730   41.624
H   1.500   21.778   38.737
H   2.983   22.421   39.415
H   1.470   23.319   39.535
H   3.231   20.348   40.933
H   2.623   20.563   42.616
H   3.875   23.137   41.770
H   4.813   21.943   41.050
H   5.987   21.802   42.981
H   4.874   20.671   43.204
H   3.441   22.881   44.158
H   4.686   24.661   45.026
H   5.697   24.136   43.632
H   6.207   23.807   45.262
H   4.803   22.030   46.532
H   5.213   20.719   45.337
H   3.206   19.851   45.885
H   2.503   21.180   44.978
H   3.044   22.408   47.595
H   2.503   20.761   47.867
H   0.479   21.401   46.195
H   0.006   23.821   46.550
H   1.478   23.711   45.658
H   1.446   24.079   47.450
H   0.717   21.833   49.162
H   -0.334   20.837   48.249
H   -0.602   22.612   48.362

