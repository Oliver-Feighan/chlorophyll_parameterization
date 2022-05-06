%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_501_chromophore_4 TDDFT with blyp functional

0 1
Mg   8.373   3.020   27.583
C   9.668   1.599   30.627
C   6.705   5.268   29.537
C   6.936   4.141   24.767
C   9.731   0.400   25.897
N   8.260   3.414   29.926
C   8.923   2.706   30.886
C   8.728   3.421   32.169
C   7.670   4.519   31.870
C   7.439   4.424   30.337
C   6.338   4.271   32.647
C   9.977   3.964   32.817
C   9.903   4.499   34.238
C   10.915   4.008   35.186
O   12.127   3.764   34.964
O   10.361   3.919   36.448
N   7.115   4.549   27.201
C   6.434   5.329   28.176
C   5.691   6.367   27.437
C   5.703   5.968   26.103
C   6.573   4.784   25.983
C   4.949   7.469   28.164
C   5.044   6.627   24.872
O   4.926   6.073   23.763
C   4.558   8.009   25.037
N   8.269   2.337   25.580
C   7.653   2.930   24.578
C   7.794   2.171   23.230
C   8.878   1.153   23.591
C   9.002   1.330   25.087
C   6.445   1.500   22.803
C   10.208   1.291   22.885
C   10.866   -0.010   22.392
N   9.538   1.414   28.130
C   10.025   0.428   27.300
C   10.881   -0.540   28.050
C   10.696   -0.063   29.332
C   9.859   1.059   29.361
C   11.787   -1.599   27.509
C   11.079   -0.333   30.710
O   11.839   -1.204   31.118
C   10.435   0.732   31.627
C   9.776   0.014   32.688
O   8.687   -0.605   32.617
O   10.415   0.230   33.871
C   9.762   -0.174   35.097
C   11.267   3.648   37.565
C   10.974   4.582   38.749
C   11.477   4.571   40.003
C   12.525   3.617   40.512
C   11.009   5.609   41.039
C   9.755   5.271   41.890
C   8.500   6.025   41.464
C   8.250   7.284   42.361
C   6.793   7.191   42.916
C   8.538   8.687   41.670
C   9.780   9.365   42.277
C   9.749   10.934   42.395
C   10.543   11.434   43.652
C   11.049   12.861   43.304
C   9.625   11.489   44.912
C   10.097   10.524   46.068
C   9.182   9.438   46.548
C   9.558   9.054   48.030
C   9.077   7.643   48.538
C   9.040   10.239   48.940
H   6.234   6.046   30.141
H   6.347   4.397   23.884
H   9.918   -0.504   25.314
H   8.288   2.654   32.805
H   8.017   5.516   32.142
H   5.845   5.031   33.254
H   6.577   3.451   33.324
H   5.539   3.816   32.061
H   10.353   4.835   32.280
H   10.840   3.323   32.639
H   8.943   4.256   34.691
H   10.210   5.542   34.165
H   3.934   7.638   27.805
H   5.459   8.432   28.149
H   4.763   7.277   29.221
H   5.363   8.608   25.463
H   3.750   7.884   25.757
H   4.379   8.392   24.033
H   8.139   2.908   22.504
H   8.612   0.104   23.464
H   6.506   0.413   22.847
H   6.198   1.732   21.767
H   5.707   1.858   23.521
H   10.936   1.826   23.494
H   10.035   2.082   22.156
H   11.478   0.257   21.531
H   10.151   -0.818   22.234
H   11.459   -0.285   23.264
H   12.839   -1.315   27.476
H   11.330   -1.913   26.571
H   11.633   -2.521   28.071
H   11.282   1.208   32.121
H   8.738   -0.527   34.972
H   9.837   0.611   35.849
H   10.279   -1.003   35.581
H   12.323   3.794   37.335
H   11.230   2.573   37.740
H   10.260   5.364   38.492
H   12.929   3.020   39.694
H   12.165   2.921   41.270
H   13.288   4.309   40.868
H   10.855   6.613   40.642
H   11.862   5.714   41.709
H   9.988   5.438   42.941
H   9.566   4.199   41.832
H   7.664   5.339   41.323
H   8.446   6.476   40.474
H   8.853   7.244   43.268
H   6.366   6.196   42.793
H   6.077   7.929   42.553
H   6.864   7.465   43.968
H   7.742   9.419   41.805
H   8.625   8.579   40.589
H   10.647   9.188   41.640
H   9.818   8.988   43.299
H   8.761   11.396   42.386
H   10.220   11.338   41.498
H   11.394   10.772   43.809
H   10.314   13.291   42.623
H   12.051   12.892   42.876
H   10.902   13.423   44.226
H   8.637   11.102   44.661
H   9.373   12.460   45.339
H   10.162   11.243   46.885
H   11.034   10.018   45.835
H   9.296   8.610   45.848
H   8.132   9.724   46.614
H   10.647   9.018   48.066
H   8.868   7.595   49.607
H   9.880   6.925   48.369
H   8.211   7.295   47.974
H   9.901   10.838   49.234
H   8.591   9.828   49.845
H   8.361   10.936   48.448

