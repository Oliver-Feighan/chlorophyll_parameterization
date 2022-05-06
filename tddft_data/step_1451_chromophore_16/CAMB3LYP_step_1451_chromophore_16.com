%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1451_chromophore_16 TDDFT with cam-b3lyp functional

0 1
Mg   40.609   41.793   27.372
C   39.721   44.017   29.976
C   41.790   39.575   29.747
C   41.734   39.752   24.887
C   39.711   44.151   25.062
N   40.548   41.721   29.639
C   40.109   42.737   30.480
C   40.376   42.267   31.968
C   41.292   41.052   31.821
C   41.240   40.746   30.328
C   42.690   41.072   32.467
C   39.083   42.088   32.855
C   38.947   43.174   33.895
C   39.806   43.162   35.146
O   40.980   43.586   35.281
O   39.091   42.622   36.255
N   41.437   39.763   27.360
C   41.801   39.010   28.427
C   42.420   37.779   27.964
C   42.444   37.902   26.533
C   41.837   39.145   26.183
C   42.864   36.636   28.971
C   43.051   36.892   25.548
O   42.868   37.025   24.313
C   43.737   35.647   25.997
N   40.919   41.971   25.381
C   41.438   41.029   24.547
C   41.425   41.531   23.102
C   40.606   42.860   23.163
C   40.327   43.000   24.651
C   42.833   41.735   22.498
C   39.443   43.005   22.200
C   38.249   41.986   22.499
N   39.742   43.700   27.427
C   39.484   44.573   26.418
C   38.941   45.815   26.961
C   38.989   45.670   28.370
C   39.556   44.349   28.582
C   38.548   46.989   26.106
C   38.764   46.268   29.676
O   38.200   47.325   30.000
C   39.288   45.251   30.764
C   40.370   45.998   31.516
O   41.408   46.350   31.040
O   40.058   46.127   32.847
C   41.168   46.614   33.785
C   39.905   42.768   37.465
C   39.126   41.945   38.499
C   38.375   42.356   39.503
C   38.103   43.837   39.698
C   37.606   41.453   40.425
C   38.407   40.342   41.199
C   37.691   38.969   41.048
C   37.804   38.059   42.264
C   37.745   36.546   41.936
C   36.691   38.552   43.242
C   37.130   38.677   44.653
C   36.665   37.408   45.434
C   35.557   37.556   46.539
C   36.265   37.913   47.873
C   34.541   36.342   46.699
C   33.048   36.721   46.763
C   32.111   35.867   47.691
C   31.453   36.637   48.848
C   30.120   36.029   49.360
C   32.505   36.719   50.018
H   42.179   38.893   30.505
H   42.154   39.182   24.056
H   39.272   44.788   24.291
H   41.030   43.005   32.432
H   40.749   40.211   32.254
H   43.284   40.340   31.919
H   42.508   40.781   33.502
H   43.148   42.045   32.643
H   39.065   41.113   33.343
H   38.230   42.134   32.179
H   37.893   43.291   34.151
H   39.360   44.069   33.430
H   42.335   35.692   28.838
H   42.943   37.006   29.993
H   43.913   36.526   28.696
H   44.186   35.104   25.165
H   43.058   34.905   26.416
H   44.513   35.949   26.701
H   40.930   40.859   22.401
H   41.220   43.731   22.936
H   42.928   41.211   21.548
H   43.550   41.307   23.198
H   42.930   42.796   22.269
H   39.743   42.882   21.159
H   38.924   43.959   22.287
H   38.461   41.173   23.193
H   38.188   41.314   21.643
H   37.274   42.435   22.689
H   38.318   46.683   25.085
H   39.485   47.543   26.068
H   37.777   47.581   26.599
H   38.547   44.987   31.519
H   42.061   45.994   33.706
H   40.796   46.437   34.794
H   41.506   47.645   33.673
H   40.159   43.784   37.770
H   40.844   42.243   37.289
H   39.315   40.872   38.445
H   37.065   44.170   39.722
H   38.595   44.512   38.998
H   38.570   44.096   40.648
H   36.709   41.147   39.886
H   37.204   42.073   41.226
H   38.526   40.690   42.225
H   39.376   40.288   40.703
H   38.047   38.427   40.171
H   36.629   39.080   40.830
H   38.783   38.237   42.709
H   37.524   36.382   40.881
H   37.038   36.006   42.566
H   38.641   35.945   42.088
H   35.884   37.819   43.252
H   36.311   39.508   42.883
H   36.724   39.587   45.095
H   38.211   38.759   44.765
H   37.508   37.068   46.037
H   36.622   36.509   44.819
H   35.027   38.491   46.357
H   35.594   37.843   48.729
H   36.531   38.970   47.845
H   37.165   37.320   48.031
H   34.784   35.621   47.480
H   34.626   35.854   45.728
H   32.504   36.803   45.822
H   33.115   37.720   47.195
H   32.536   34.939   48.075
H   31.236   35.626   47.088
H   31.330   37.658   48.485
H   30.123   34.976   49.078
H   29.352   36.499   48.746
H   29.983   36.267   50.414
H   32.210   36.000   50.782
H   32.424   37.645   50.586
H   33.560   36.680   49.745

