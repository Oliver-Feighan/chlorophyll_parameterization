%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1751_chromophore_10 TDDFT with cam-b3lyp functional

0 1
Mg   41.571   8.087   29.361
C   43.212   9.374   32.181
C   39.454   6.551   31.676
C   40.147   6.389   26.813
C   43.873   9.257   27.347
N   41.506   7.874   31.686
C   42.168   8.682   32.617
C   41.488   8.707   33.987
C   40.374   7.606   33.784
C   40.416   7.253   32.301
C   40.401   6.439   34.776
C   40.969   10.125   34.484
C   41.841   10.700   35.597
C   41.148   11.327   36.790
O   40.133   12.012   36.701
O   41.757   11.016   37.962
N   39.885   6.829   29.297
C   39.124   6.341   30.293
C   37.997   5.652   29.771
C   38.183   5.587   28.384
C   39.458   6.323   28.102
C   36.929   4.974   30.683
C   37.137   5.018   27.373
O   37.388   5.015   26.160
C   35.723   4.550   27.796
N   41.890   7.958   27.283
C   41.272   7.131   26.418
C   41.780   7.338   24.991
C   42.821   8.473   25.151
C   42.876   8.613   26.632
C   42.321   6.060   24.301
C   42.459   9.779   24.499
C   41.471   10.768   25.146
N   43.182   9.140   29.612
C   44.060   9.559   28.678
C   45.116   10.248   29.340
C   44.881   10.297   30.712
C   43.654   9.523   30.819
C   46.359   10.844   28.696
C   45.276   10.740   32.038
O   46.221   11.425   32.413
C   44.094   10.236   33.032
C   44.778   9.483   34.190
O   45.139   8.296   34.153
O   44.828   10.256   35.351
C   45.376   9.528   36.570
C   41.354   11.840   39.109
C   40.657   11.101   40.198
C   40.560   11.599   41.448
C   41.267   12.852   41.991
C   39.891   10.729   42.540
C   38.367   10.635   42.298
C   37.690   9.895   43.535
C   37.020   10.748   44.633
C   37.529   10.488   46.020
C   35.467   10.609   44.459
C   34.729   11.905   44.889
C   33.730   11.928   46.037
C   34.263   12.703   47.303
C   34.546   11.783   48.585
C   33.219   13.820   47.671
C   33.429   15.146   46.993
C   32.215   15.713   46.295
C   31.981   17.243   46.503
C   31.045   17.518   47.663
C   31.627   17.835   45.064
H   38.749   6.198   32.432
H   39.717   5.871   25.954
H   44.625   9.542   26.608
H   42.263   8.335   34.658
H   39.428   8.139   33.882
H   39.412   6.335   35.222
H   40.938   6.749   35.672
H   40.849   5.513   34.416
H   39.912   10.169   34.744
H   41.129   10.885   33.719
H   42.569   11.369   35.138
H   42.536   9.944   35.962
H   35.931   5.333   30.433
H   37.120   5.130   31.745
H   36.965   3.900   30.498
H   35.134   4.192   26.952
H   35.316   5.462   28.233
H   35.929   3.771   28.531
H   40.967   7.706   24.364
H   43.831   8.293   24.784
H   41.736   5.790   23.422
H   42.159   5.249   25.011
H   43.352   6.289   24.030
H   42.128   9.622   23.473
H   43.402   10.317   24.402
H   41.823   11.519   25.853
H   40.800   10.263   25.842
H   40.987   11.417   24.417
H   46.375   11.912   28.915
H   46.391   10.691   27.617
H   47.280   10.341   28.990
H   43.509   11.097   33.355
H   44.553   9.341   37.259
H   46.048   10.171   37.139
H   45.903   8.620   36.278
H   40.747   12.727   38.924
H   42.297   12.312   39.386
H   40.158   10.151   40.003
H   40.494   13.548   42.317
H   41.901   13.380   41.279
H   41.792   12.622   42.918
H   39.978   11.186   43.526
H   40.441   9.789   42.510
H   38.092   10.127   41.374
H   37.983   11.647   42.172
H   38.432   9.293   44.060
H   37.007   9.233   43.001
H   37.275   11.744   44.272
H   38.586   10.224   46.037
H   37.083   9.595   46.458
H   37.265   11.327   46.665
H   35.063   9.776   45.035
H   35.191   10.326   43.443
H   34.158   12.280   44.040
H   35.588   12.543   45.098
H   33.457   10.907   46.301
H   32.817   12.432   45.720
H   35.190   13.218   47.048
H   35.407   12.182   49.120
H   34.703   10.749   48.279
H   33.704   11.756   49.277
H   33.216   14.024   48.742
H   32.206   13.469   47.475
H   34.082   14.964   46.140
H   34.050   15.764   47.641
H   31.290   15.231   46.612
H   32.394   15.454   45.251
H   32.953   17.619   46.822
H   31.554   17.772   48.593
H   30.430   16.648   47.890
H   30.389   18.349   47.404
H   32.383   18.618   45.017
H   30.626   18.267   45.033
H   31.756   17.078   44.291

