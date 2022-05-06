%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_651_chromophore_6 TDDFT with PBE1PBE functional

0 1
Mg   16.905   -2.701   27.622
C   16.456   -0.355   30.224
C   18.746   -4.555   29.695
C   18.362   -4.186   24.898
C   15.770   -0.136   25.414
N   17.606   -2.382   29.634
C   17.192   -1.444   30.605
C   17.682   -1.809   32.025
C   18.635   -3.034   31.736
C   18.243   -3.398   30.270
C   20.227   -2.862   32.018
C   16.427   -2.192   32.964
C   16.399   -1.869   34.453
C   17.813   -1.454   35.062
O   18.133   -0.279   35.261
O   18.581   -2.580   35.474
N   18.120   -4.378   27.392
C   18.669   -5.020   28.371
C   19.340   -6.171   27.844
C   19.456   -5.943   26.417
C   18.579   -4.764   26.155
C   19.942   -7.273   28.647
C   20.186   -6.792   25.296
O   20.108   -6.531   24.125
C   21.063   -7.971   25.791
N   16.968   -2.218   25.504
C   17.609   -3.032   24.576
C   17.667   -2.444   23.189
C   16.648   -1.283   23.304
C   16.439   -1.171   24.831
C   19.075   -1.861   22.944
C   15.306   -1.458   22.565
C   14.889   -0.263   21.688
N   16.026   -0.835   27.755
C   15.522   0.047   26.811
C   14.910   1.163   27.406
C   15.285   1.102   28.745
C   15.965   -0.161   28.891
C   14.092   2.235   26.714
C   15.139   1.800   29.965
O   14.479   2.820   30.167
C   15.751   0.817   30.999
C   16.678   1.646   31.814
O   17.895   1.587   31.823
O   15.968   2.492   32.627
C   16.763   3.398   33.421
C   20.006   -2.475   36.010
C   20.211   -3.365   37.287
C   19.808   -3.085   38.539
C   18.805   -1.995   38.763
C   20.444   -3.931   39.706
C   20.500   -5.479   39.502
C   21.360   -6.284   40.491
C   22.434   -7.066   39.717
C   21.995   -8.315   39.089
C   23.777   -7.206   40.506
C   23.836   -8.208   41.693
C   25.085   -8.038   42.563
C   26.039   -9.196   42.577
C   25.582   -10.408   43.380
C   27.396   -8.751   43.014
C   28.451   -9.784   42.546
C   29.509   -9.106   41.608
C   30.802   -10.004   41.540
C   31.041   -10.761   40.208
C   32.120   -9.243   41.722
H   19.315   -5.205   30.363
H   18.739   -4.706   24.015
H   15.330   0.560   24.697
H   18.319   -1.020   32.424
H   18.467   -3.843   32.447
H   20.673   -3.767   32.431
H   20.370   -2.050   32.731
H   20.688   -2.548   31.081
H   16.108   -3.227   32.836
H   15.567   -1.736   32.474
H   15.937   -2.678   35.019
H   15.782   -0.995   34.663
H   19.398   -8.180   28.386
H   19.636   -7.180   29.689
H   21.032   -7.308   28.622
H   20.597   -8.649   26.506
H   21.836   -7.471   26.372
H   21.481   -8.520   24.947
H   17.449   -3.175   22.411
H   16.911   -0.278   22.973
H   19.779   -2.244   23.682
H   19.140   -0.779   23.059
H   19.442   -2.143   21.957
H   14.540   -1.592   23.329
H   15.429   -2.361   21.966
H   13.926   -0.530   21.253
H   15.658   0.047   20.980
H   14.758   0.652   22.267
H   14.783   2.719   26.023
H   13.700   2.964   27.423
H   13.331   1.767   26.089
H   14.936   0.511   31.655
H   16.162   4.233   33.782
H   17.489   3.928   32.804
H   17.147   2.792   34.241
H   20.133   -1.433   36.301
H   20.726   -2.602   35.202
H   21.048   -4.058   37.191
H   18.794   -1.847   39.843
H   17.776   -2.193   38.463
H   19.116   -1.078   38.263
H   20.013   -3.697   40.679
H   21.504   -3.679   39.729
H   20.733   -5.691   38.459
H   19.451   -5.755   39.608
H   20.714   -6.957   41.055
H   21.822   -5.563   41.165
H   22.763   -6.510   38.839
H   22.063   -8.232   38.004
H   21.145   -8.854   39.507
H   22.784   -9.054   39.222
H   24.048   -6.181   40.758
H   24.468   -7.395   39.685
H   23.920   -9.180   41.208
H   23.060   -8.062   42.444
H   24.711   -7.876   43.574
H   25.642   -7.150   42.263
H   26.099   -9.444   41.517
H   26.248   -10.812   44.142
H   25.363   -11.143   42.606
H   24.617   -10.179   43.831
H   27.527   -8.524   44.072
H   27.665   -7.917   42.366
H   27.879   -10.611   42.124
H   29.005   -10.178   43.398
H   29.674   -8.032   41.686
H   29.082   -9.168   40.607
H   30.661   -10.740   42.331
H   31.533   -10.218   39.401
H   30.046   -10.833   39.769
H   31.538   -11.721   40.352
H   32.283   -9.196   42.798
H   31.956   -8.236   41.340
H   32.931   -9.779   41.228

