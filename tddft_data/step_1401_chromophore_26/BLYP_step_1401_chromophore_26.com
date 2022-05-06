%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1401_chromophore_26 TDDFT with blyp functional

0 1
Mg   -9.409   18.685   42.498
C   -6.039   17.996   42.581
C   -8.705   22.050   41.842
C   -12.591   19.317   42.389
C   -9.986   15.208   42.483
N   -7.581   19.793   41.971
C   -6.302   19.263   42.201
C   -5.244   20.367   42.131
C   -6.137   21.697   42.117
C   -7.556   21.158   41.987
C   -5.998   22.581   43.341
C   -4.151   20.268   40.982
C   -4.420   19.550   39.676
C   -3.986   20.193   38.332
O   -2.862   20.226   37.853
O   -5.094   20.736   37.716
N   -10.537   20.407   42.027
C   -10.047   21.664   41.843
C   -11.131   22.578   41.625
C   -12.306   21.814   41.864
C   -11.869   20.502   42.105
C   -10.993   24.046   41.267
C   -13.703   22.349   41.748
O   -14.009   23.500   41.350
C   -14.861   21.448   41.883
N   -11.074   17.414   42.246
C   -12.292   17.908   42.480
C   -13.337   16.861   42.826
C   -12.500   15.488   42.790
C   -11.092   16.059   42.432
C   -13.954   17.216   44.234
C   -12.984   14.434   41.805
C   -13.728   13.259   42.494
N   -8.258   16.935   42.491
C   -8.647   15.628   42.659
C   -7.491   14.830   42.912
C   -6.448   15.730   42.925
C   -6.971   16.983   42.647
C   -7.458   13.353   43.237
C   -5.023   15.844   43.156
O   -4.171   15.008   43.340
C   -4.653   17.349   42.954
C   -4.119   17.908   44.227
O   -2.982   18.328   44.360
O   -5.120   18.066   45.134
C   -4.719   18.749   46.337
C   -4.805   21.330   36.404
C   -5.774   20.856   35.272
C   -6.106   19.621   34.859
C   -5.821   18.270   35.471
C   -6.835   19.504   33.505
C   -8.337   19.488   33.588
C   -9.073   20.407   32.588
C   -10.038   21.371   33.219
C   -11.294   20.606   33.796
C   -10.492   22.542   32.320
C   -10.168   23.906   32.936
C   -9.585   24.958   32.058
C   -10.628   25.499   31.099
C   -10.859   26.983   31.374
C   -10.185   25.227   29.595
C   -11.341   24.633   28.858
C   -11.398   24.973   27.361
C   -12.286   23.993   26.597
C   -11.489   22.825   25.948
C   -13.259   24.743   25.644
H   -8.592   23.135   41.803
H   -13.665   19.489   42.490
H   -10.263   14.159   42.607
H   -4.712   20.342   43.082
H   -5.782   22.341   41.312
H   -6.893   22.860   43.896
H   -5.430   23.477   43.088
H   -5.323   22.171   44.092
H   -3.218   19.875   41.384
H   -3.723   21.247   40.765
H   -5.494   19.382   39.591
H   -3.847   18.625   39.732
H   -10.238   24.540   41.878
H   -11.845   24.726   41.231
H   -10.608   24.119   40.250
H   -14.994   21.117   42.913
H   -14.804   20.670   41.122
H   -15.775   22.028   41.749
H   -14.096   16.838   42.044
H   -12.446   15.094   43.804
H   -14.777   17.903   44.035
H   -13.240   17.688   44.909
H   -14.428   16.324   44.642
H   -12.047   14.060   41.392
H   -13.610   14.681   40.947
H   -13.150   12.340   42.588
H   -14.662   13.051   41.972
H   -13.880   13.409   43.562
H   -6.469   13.099   42.856
H   -8.323   12.820   42.842
H   -7.411   13.215   44.317
H   -3.816   17.653   42.325
H   -5.600   18.899   46.960
H   -4.260   19.712   46.117
H   -3.930   18.200   46.852
H   -3.816   21.227   35.957
H   -4.985   22.404   36.361
H   -6.117   21.566   34.520
H   -5.559   18.447   36.515
H   -5.050   17.679   34.977
H   -6.749   17.711   35.356
H   -6.653   18.517   33.080
H   -6.472   20.161   32.714
H   -8.623   19.807   34.590
H   -8.701   18.474   33.425
H   -9.577   19.857   31.794
H   -8.283   21.020   32.155
H   -9.579   21.736   34.138
H   -12.131   20.716   33.107
H   -11.687   21.006   34.731
H   -11.057   19.563   34.004
H   -11.578   22.528   32.228
H   -9.924   22.550   31.390
H   -9.461   23.813   33.761
H   -11.054   24.287   33.443
H   -8.701   24.472   31.644
H   -9.293   25.722   32.779
H   -11.605   25.152   31.433
H   -10.044   27.405   31.963
H   -11.795   27.132   31.912
H   -11.009   27.575   30.471
H   -9.307   24.581   29.594
H   -9.944   26.201   29.168
H   -12.384   24.846   29.093
H   -11.349   23.566   29.080
H   -10.381   24.834   26.994
H   -11.786   25.990   27.308
H   -12.848   23.448   27.355
H   -12.201   22.001   25.915
H   -10.554   22.623   26.471
H   -11.223   22.962   24.900
H   -14.101   25.168   26.190
H   -13.804   24.128   24.928
H   -12.889   25.479   24.930
