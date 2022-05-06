%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1051_chromophore_26 TDDFT with PBE1PBE functional

0 1
Mg   -8.885   18.532   42.779
C   -5.328   18.486   42.745
C   -8.674   21.982   42.054
C   -12.240   18.600   41.940
C   -8.868   15.038   42.917
N   -7.262   19.951   42.492
C   -5.883   19.721   42.439
C   -5.063   20.928   42.174
C   -6.091   22.100   42.207
C   -7.448   21.331   42.303
C   -5.969   23.175   43.436
C   -4.127   20.727   40.889
C   -4.780   20.153   39.603
C   -4.424   20.834   38.267
O   -3.349   21.382   37.999
O   -5.458   20.765   37.349
N   -10.326   20.039   42.111
C   -10.039   21.403   41.834
C   -11.235   22.107   41.457
C   -12.275   21.134   41.293
C   -11.688   19.866   41.819
C   -11.263   23.589   41.226
C   -13.651   21.366   40.716
O   -14.033   22.456   40.427
C   -14.499   20.240   40.607
N   -10.350   17.078   42.573
C   -11.668   17.312   42.272
C   -12.437   16.050   42.144
C   -11.393   14.957   42.520
C   -10.076   15.735   42.586
C   -13.721   16.003   43.019
C   -11.435   13.864   41.353
C   -11.908   12.503   41.943
N   -7.463   17.012   42.869
C   -7.604   15.637   43.036
C   -6.255   15.101   43.259
C   -5.345   16.194   43.125
C   -6.145   17.351   42.882
C   -5.896   13.707   43.564
C   -3.938   16.454   43.179
O   -2.948   15.749   43.383
C   -3.862   17.984   42.890
C   -3.332   18.645   44.129
O   -2.341   19.364   44.165
O   -3.983   18.195   45.243
C   -3.704   19.020   46.407
C   -5.247   21.216   35.954
C   -6.232   20.783   34.930
C   -6.430   19.494   34.542
C   -5.765   18.240   35.124
C   -7.330   19.143   33.381
C   -8.761   18.573   33.487
C   -9.539   18.668   32.155
C   -11.016   19.152   32.505
C   -11.936   17.941   32.466
C   -11.430   20.161   31.435
C   -11.101   21.643   31.811
C   -10.298   22.502   30.783
C   -11.053   23.767   30.195
C   -10.268   25.094   30.688
C   -10.972   23.584   28.648
C   -12.291   22.935   28.097
C   -12.066   22.169   26.766
C   -12.831   22.684   25.563
C   -12.935   21.679   24.513
C   -12.100   24.017   25.024
H   -8.504   23.046   41.876
H   -13.281   18.429   41.657
H   -9.038   13.964   43.020
H   -4.404   20.986   43.041
H   -6.095   22.482   41.186
H   -6.950   23.385   43.861
H   -5.518   24.078   43.025
H   -5.361   22.854   44.282
H   -3.210   20.240   41.222
H   -3.797   21.709   40.548
H   -5.866   20.217   39.529
H   -4.396   19.138   39.499
H   -12.031   23.787   40.479
H   -10.288   23.849   40.814
H   -11.434   24.217   42.100
H   -14.972   19.826   41.497
H   -13.967   19.512   39.995
H   -15.330   20.353   39.910
H   -12.818   16.016   41.123
H   -11.707   14.499   43.458
H   -13.583   16.638   43.894
H   -13.983   15.032   43.440
H   -14.488   16.215   42.274
H   -10.413   13.748   40.989
H   -11.977   14.074   40.431
H   -12.224   11.953   41.056
H   -12.762   12.679   42.597
H   -11.109   11.999   42.487
H   -4.807   13.706   43.594
H   -6.236   13.117   42.713
H   -6.376   13.432   44.503
H   -3.309   18.192   41.974
H   -4.472   18.764   47.137
H   -3.769   20.083   46.172
H   -2.746   18.926   46.918
H   -4.423   20.621   35.561
H   -5.071   22.291   35.938
H   -6.742   21.609   34.434
H   -6.460   17.506   35.532
H   -5.150   18.481   35.991
H   -5.169   17.673   34.410
H   -6.919   18.316   32.802
H   -7.271   19.966   32.669
H   -9.284   19.085   34.295
H   -8.626   17.528   33.764
H   -9.514   17.671   31.716
H   -9.014   19.393   31.532
H   -11.023   19.610   33.494
H   -11.319   17.047   32.544
H   -12.535   17.778   31.570
H   -12.548   18.000   33.367
H   -12.501   20.069   31.256
H   -11.012   19.837   30.483
H   -10.691   21.586   32.820
H   -12.026   22.176   32.032
H   -10.172   21.742   30.011
H   -9.337   22.823   31.184
H   -12.080   23.752   30.560
H   -11.089   25.755   30.962
H   -9.692   25.585   29.904
H   -9.633   24.770   31.513
H   -10.055   23.071   28.356
H   -10.880   24.611   28.294
H   -13.112   23.652   28.075
H   -12.574   22.291   28.929
H   -12.364   21.179   27.112
H   -11.029   22.224   26.436
H   -13.866   22.799   25.884
H   -13.977   21.544   24.222
H   -12.494   20.726   24.807
H   -12.333   21.788   23.611
H   -11.729   23.925   24.003
H   -11.226   24.168   25.657
H   -12.782   24.847   25.207
