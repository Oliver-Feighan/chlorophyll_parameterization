%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1551_chromophore_3 TDDFT with cam-b3lyp functional

0 1
Mg   1.788   7.995   25.939
C   2.321   9.897   28.840
C   1.563   5.197   27.944
C   1.640   6.176   23.147
C   2.467   10.872   23.996
N   1.941   7.598   28.086
C   2.168   8.538   29.131
C   2.141   7.855   30.584
C   2.224   6.315   30.216
C   1.904   6.329   28.630
C   3.504   5.527   30.663
C   0.915   8.333   31.414
C   1.132   8.626   32.904
C   2.354   8.008   33.646
O   3.415   8.626   33.792
O   2.063   6.732   33.898
N   1.528   6.010   25.557
C   1.416   5.039   26.529
C   1.037   3.840   25.825
C   1.017   4.045   24.461
C   1.399   5.471   24.328
C   0.587   2.648   26.523
C   0.665   3.133   23.267
O   0.588   3.589   22.123
C   0.396   1.637   23.417
N   2.190   8.456   23.929
C   1.868   7.496   22.930
C   1.878   8.085   21.498
C   1.857   9.638   21.786
C   2.297   9.654   23.280
C   3.134   7.583   20.675
C   0.519   10.339   21.679
C   0.483   11.641   20.803
N   2.234   10.075   26.315
C   2.526   11.045   25.432
C   2.688   12.267   26.177
C   2.606   11.909   27.516
C   2.308   10.510   27.573
C   2.889   13.691   25.666
C   2.711   12.345   28.870
O   2.911   13.482   29.272
C   2.452   11.022   29.784
C   3.554   11.012   30.667
O   4.600   10.333   30.621
O   3.311   11.932   31.635
C   4.417   12.317   32.498
C   3.162   5.986   34.451
C   2.652   4.624   34.892
C   2.190   4.230   36.081
C   1.991   5.168   37.287
C   1.518   2.919   36.191
C   1.683   2.131   37.470
C   1.793   0.614   37.212
C   1.378   -0.174   38.534
C   -0.102   -0.690   38.405
C   2.381   -1.290   39.054
C   3.037   -1.074   40.470
C   4.541   -1.357   40.514
C   5.053   -2.754   40.894
C   6.597   -2.672   41.232
C   4.684   -3.833   39.825
C   3.509   -4.668   40.260
C   3.955   -6.088   40.543
C   3.136   -7.187   39.861
C   2.033   -7.679   40.819
C   3.984   -8.443   39.365
H   1.233   4.365   28.570
H   1.586   5.642   22.196
H   2.441   11.767   23.371
H   3.051   8.014   31.161
H   1.419   5.796   30.735
H   3.166   4.912   31.497
H   4.161   6.343   30.964
H   3.992   4.858   29.955
H   0.069   7.671   31.227
H   0.550   9.272   30.998
H   0.260   8.204   33.403
H   1.247   9.676   33.174
H   1.387   1.912   26.605
H   -0.308   2.225   26.068
H   0.279   2.839   27.551
H   1.036   1.127   24.137
H   0.549   1.090   22.487
H   -0.571   1.523   23.906
H   0.995   7.724   20.970
H   2.524   10.270   21.200
H   2.909   7.141   19.704
H   3.758   6.846   21.180
H   3.786   8.428   20.454
H   0.243   10.834   22.611
H   -0.264   9.650   21.362
H   -0.295   11.499   20.054
H   1.393   11.925   20.274
H   0.161   12.533   21.340
H   3.386   13.727   24.697
H   3.498   14.335   26.300
H   1.887   14.121   25.640
H   1.583   11.332   30.364
H   5.405   12.267   32.041
H   4.399   11.535   33.257
H   4.243   13.303   32.929
H   3.749   6.340   35.299
H   3.826   5.646   33.657
H   2.731   3.929   34.056
H   2.041   6.194   36.921
H   2.760   4.997   38.040
H   1.000   5.004   37.709
H   1.617   2.367   35.256
H   0.437   3.040   36.261
H   0.984   2.311   38.287
H   2.570   2.397   38.046
H   2.854   0.371   37.159
H   1.266   0.270   36.322
H   1.192   0.517   39.356
H   -0.715   -0.092   39.080
H   -0.356   -1.663   38.826
H   -0.595   -0.472   37.458
H   3.078   -1.577   38.267
H   1.808   -2.199   39.238
H   2.509   -1.726   41.166
H   2.838   -0.037   40.739
H   5.004   -0.601   41.148
H   4.885   -1.149   39.500
H   4.542   -2.952   41.836
H   6.846   -2.831   42.281
H   6.906   -1.640   41.070
H   7.146   -3.411   40.648
H   5.544   -4.470   39.622
H   4.475   -3.415   38.840
H   2.718   -4.757   39.515
H   3.130   -4.285   41.207
H   3.819   -6.201   41.619
H   5.007   -6.304   40.360
H   2.580   -6.747   39.033
H   1.996   -8.759   40.964
H   1.076   -7.398   40.378
H   2.020   -7.148   41.771
H   3.396   -9.334   39.587
H   4.859   -8.412   40.014
H   4.251   -8.434   38.308
