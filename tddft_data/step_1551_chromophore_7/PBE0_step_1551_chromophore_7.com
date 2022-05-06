%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1551_chromophore_7 TDDFT with PBE1PBE functional

0 1
Mg   25.968   -0.330   29.778
C   27.920   -0.780   32.497
C   23.168   -0.054   31.647
C   24.246   -0.221   26.807
C   28.864   -0.940   27.695
N   25.711   -0.323   31.846
C   26.593   -0.554   32.810
C   25.979   -0.285   34.169
C   24.438   -0.187   33.904
C   24.384   -0.203   32.355
C   23.394   -1.198   34.545
C   26.669   0.963   34.899
C   25.841   1.663   36.051
C   26.771   1.909   37.315
O   27.668   2.748   37.464
O   26.511   0.979   38.266
N   23.967   -0.193   29.308
C   22.977   -0.048   30.222
C   21.736   0.149   29.460
C   22.030   0.208   28.086
C   23.467   -0.020   28.007
C   20.463   0.215   30.088
C   21.147   0.464   26.932
O   21.625   0.778   25.860
C   19.660   0.484   27.116
N   26.487   -0.815   27.543
C   25.582   -0.573   26.543
C   26.194   -0.272   25.179
C   27.723   -0.543   25.453
C   27.696   -0.792   26.974
C   25.755   -1.299   24.170
C   28.587   0.738   25.159
C   30.040   0.581   24.708
N   27.988   -0.666   30.009
C   29.058   -0.918   29.107
C   30.285   -1.257   29.947
C   29.893   -1.167   31.209
C   28.516   -0.822   31.238
C   31.574   -1.612   29.370
C   30.382   -1.242   32.529
O   31.473   -1.358   33.062
C   29.108   -0.986   33.512
C   28.869   -2.074   34.417
O   28.074   -2.984   34.223
O   29.514   -1.818   35.600
C   29.275   -2.847   36.573
C   27.267   1.099   39.543
C   26.340   1.341   40.715
C   26.631   1.502   42.031
C   28.009   1.783   42.674
C   25.438   1.827   43.076
C   25.126   3.356   43.099
C   23.607   3.506   43.270
C   23.179   3.852   44.763
C   22.353   2.757   45.383
C   22.462   5.186   44.900
C   22.632   5.861   46.289
C   21.785   7.138   46.373
C   22.483   8.205   47.217
C   22.836   7.663   48.638
C   21.719   9.558   47.343
C   22.677   10.693   46.883
C   22.820   10.778   45.308
C   22.606   12.188   44.675
C   22.048   12.215   43.218
C   23.732   13.114   44.790
H   22.329   0.139   32.320
H   23.693   -0.141   25.870
H   29.718   -1.066   27.026
H   26.170   -1.202   34.727
H   24.072   0.816   34.124
H   23.133   -1.797   33.672
H   22.529   -0.669   34.945
H   23.994   -1.716   35.293
H   26.932   1.653   34.098
H   27.597   0.620   35.357
H   24.994   1.031   36.318
H   25.434   2.600   35.671
H   20.418   -0.043   31.146
H   19.813   -0.455   29.525
H   20.118   1.248   30.115
H   19.163   -0.475   27.265
H   19.252   0.947   26.218
H   19.478   1.172   27.942
H   26.072   0.747   24.811
H   28.080   -1.473   25.009
H   26.513   -1.762   23.537
H   25.015   -0.804   23.541
H   25.083   -2.062   24.562
H   28.607   1.346   26.062
H   28.043   1.217   24.345
H   30.300   -0.477   24.680
H   30.722   1.207   25.283
H   30.162   0.821   23.652
H   32.322   -1.187   30.040
H   31.743   -1.223   28.367
H   31.816   -2.674   29.424
H   29.178   -0.066   34.092
H   30.170   -2.766   37.190
H   29.307   -3.860   36.172
H   28.367   -2.769   37.171
H   27.965   1.929   39.646
H   27.887   0.204   39.602
H   25.309   1.564   40.440
H   28.813   1.541   41.979
H   27.980   1.234   43.615
H   28.154   2.819   42.981
H   25.814   1.544   44.059
H   24.592   1.198   42.798
H   25.435   3.916   42.216
H   25.669   3.797   43.935
H   22.995   2.671   42.928
H   23.448   4.420   42.698
H   24.119   3.908   45.311
H   22.520   1.773   44.946
H   21.294   2.999   45.287
H   22.641   2.490   46.400
H   21.399   5.201   44.657
H   22.840   5.876   44.146
H   23.697   6.065   46.401
H   22.320   5.082   46.985
H   20.815   6.966   46.839
H   21.689   7.515   45.354
H   23.420   8.430   46.708
H   23.844   7.252   48.587
H   22.214   6.823   48.947
H   22.905   8.460   49.379
H   21.306   9.821   48.317
H   20.804   9.586   46.752
H   23.605   10.462   47.406
H   22.327   11.676   47.200
H   22.303   10.013   44.728
H   23.864   10.490   45.187
H   21.797   12.609   45.272
H   22.905   12.377   42.565
H   21.193   12.888   43.159
H   21.736   11.193   43.001
H   23.809   13.626   45.749
H   23.753   13.872   44.007
H   24.686   12.601   44.660

