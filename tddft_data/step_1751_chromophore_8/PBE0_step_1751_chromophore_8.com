%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1751_chromophore_8 TDDFT with PBE1PBE functional

0 1
Mg   44.142   3.315   48.038
C   42.358   6.271   47.610
C   41.337   1.561   47.872
C   46.011   0.561   47.682
C   47.157   5.241   47.637
N   42.118   3.923   47.666
C   41.522   5.119   47.706
C   40.025   5.063   47.851
C   39.813   3.534   48.179
C   41.183   2.947   47.944
C   39.271   3.303   49.637
C   39.308   5.522   46.505
C   39.800   5.046   45.102
C   39.755   6.026   43.906
O   39.138   7.082   43.863
O   40.399   5.487   42.800
N   43.695   1.321   47.738
C   42.469   0.777   47.692
C   42.579   -0.569   47.473
C   43.941   -0.964   47.433
C   44.649   0.309   47.665
C   41.339   -1.398   47.332
C   44.486   -2.366   47.285
O   43.712   -3.298   47.101
C   45.921   -2.575   47.168
N   46.338   2.985   47.704
C   46.814   1.745   47.781
C   48.318   1.650   48.023
C   48.681   3.217   48.052
C   47.329   3.841   47.681
C   48.691   0.924   49.339
C   49.768   3.686   47.002
C   51.184   4.029   47.532
N   44.684   5.434   47.894
C   45.918   5.992   47.676
C   45.764   7.415   47.491
C   44.373   7.602   47.513
C   43.769   6.321   47.696
C   46.784   8.414   47.208
C   43.399   8.573   47.368
O   43.446   9.769   47.179
C   42.010   7.755   47.332
C   41.238   8.365   48.410
O   41.587   8.330   49.599
O   40.065   8.850   47.906
C   39.029   9.391   48.808
C   40.697   6.350   41.676
C   41.397   5.629   40.577
C   42.034   6.052   39.531
C   42.029   7.458   39.027
C   42.610   5.011   38.530
C   41.566   4.165   37.710
C   41.495   2.645   38.055
C   41.446   1.867   36.751
C   40.639   0.560   36.965
C   42.901   1.560   36.335
C   43.191   1.931   34.784
C   44.164   3.071   34.683
C   44.637   3.349   33.281
C   46.195   3.350   33.210
C   44.277   4.706   32.647
C   43.262   4.620   31.400
C   43.709   5.596   30.294
C   44.386   4.866   29.180
C   43.603   4.289   27.994
C   45.375   5.806   28.505
H   40.335   1.135   47.955
H   46.702   -0.273   47.542
H   48.123   5.746   47.575
H   39.603   5.700   48.629
H   39.076   3.105   47.500
H   40.090   2.832   50.181
H   38.402   2.659   49.504
H   39.034   4.216   50.183
H   39.591   6.575   46.492
H   38.222   5.453   46.560
H   39.070   4.250   44.960
H   40.787   4.586   45.161
H   40.377   -1.072   47.727
H   41.629   -2.202   48.009
H   41.275   -1.973   46.408
H   46.300   -2.481   48.186
H   46.539   -1.968   46.507
H   46.154   -3.615   46.940
H   48.697   1.094   47.165
H   48.993   3.439   49.073
H   48.872   -0.118   49.075
H   47.916   1.002   50.101
H   49.627   1.381   49.663
H   49.404   4.414   46.278
H   49.901   2.775   46.419
H   51.064   3.940   48.611
H   51.485   5.069   47.403
H   51.934   3.432   47.014
H   47.692   7.927   46.852
H   46.948   8.871   48.185
H   46.345   9.076   46.462
H   41.562   7.969   46.361
H   39.104   9.147   49.868
H   38.059   8.976   48.535
H   38.740   10.427   48.628
H   41.437   7.070   42.025
H   39.815   6.843   41.266
H   41.436   4.553   40.747
H   43.034   7.827   38.821
H   41.505   8.191   39.639
H   41.526   7.426   38.061
H   43.232   4.361   39.146
H   43.459   5.412   37.978
H   41.814   4.360   36.667
H   40.597   4.633   37.881
H   40.619   2.442   38.671
H   42.351   2.407   38.687
H   41.068   2.444   35.907
H   39.659   1.027   36.859
H   40.795   0.242   37.996
H   40.686   -0.194   36.179
H   43.279   0.575   36.609
H   43.587   2.178   36.913
H   42.261   2.159   34.263
H   43.623   1.063   34.287
H   44.994   3.004   35.386
H   43.632   3.946   35.057
H   44.267   2.640   32.541
H   46.582   4.182   33.798
H   46.569   3.453   32.191
H   46.675   2.452   33.599
H   45.200   5.243   32.431
H   43.792   5.398   33.336
H   42.274   4.954   31.717
H   43.107   3.596   31.059
H   44.397   6.349   30.680
H   42.935   6.214   29.839
H   44.869   4.002   29.637
H   43.366   5.108   27.315
H   42.666   3.902   28.395
H   44.286   3.571   27.540
H   45.452   5.787   27.418
H   46.375   5.645   28.906
H   45.196   6.862   28.707

