%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1701_chromophore_11 TDDFT with cam-b3lyp functional

0 1
Mg   53.350   23.995   45.507
C   50.567   26.219   44.656
C   51.249   21.302   44.781
C   55.992   22.052   45.651
C   55.281   26.864   45.925
N   51.290   23.855   44.599
C   50.285   24.860   44.519
C   48.859   24.214   44.432
C   49.119   22.769   44.876
C   50.658   22.599   44.721
C   48.737   22.552   46.417
C   48.164   24.408   43.074
C   49.027   24.289   41.799
C   48.335   23.629   40.490
O   47.894   22.480   40.375
O   48.286   24.480   39.408
N   53.551   21.905   45.287
C   52.625   21.002   44.904
C   53.294   19.695   44.857
C   54.690   19.891   44.863
C   54.814   21.334   45.292
C   52.536   18.427   44.528
C   55.743   18.812   44.619
O   55.498   17.808   43.945
C   57.196   18.982   44.986
N   55.373   24.449   45.551
C   56.267   23.402   45.768
C   57.742   24.043   45.916
C   57.488   25.610   45.965
C   55.995   25.708   45.798
C   58.708   23.404   46.915
C   58.315   26.436   44.949
C   57.513   26.929   43.745
N   53.060   26.068   45.374
C   53.912   27.108   45.604
C   53.319   28.436   45.409
C   51.921   28.070   45.118
C   51.864   26.658   44.958
C   53.997   29.771   45.628
C   50.628   28.628   44.946
O   50.268   29.805   44.863
C   49.766   27.506   44.622
C   49.259   27.858   43.286
O   49.946   27.834   42.237
O   47.933   28.179   43.331
C   47.372   28.810   42.173
C   47.699   23.958   38.172
C   48.812   23.498   37.258
C   49.025   22.194   36.844
C   48.165   20.984   37.078
C   50.143   21.914   35.858
C   51.108   20.714   36.041
C   51.369   19.963   34.724
C   52.886   19.424   34.727
C   53.706   19.984   33.557
C   52.963   17.917   34.969
C   52.147   17.181   33.911
C   52.799   15.835   33.476
C   51.925   14.943   32.549
C   51.169   13.871   33.331
C   52.792   14.254   31.481
C   52.787   14.947   30.065
C   54.261   15.224   29.555
C   54.528   16.785   29.234
C   55.397   16.779   27.891
C   55.230   17.465   30.376
H   50.660   20.383   44.746
H   56.899   21.446   45.720
H   55.903   27.703   46.244
H   48.153   24.733   45.080
H   48.569   22.106   44.208
H   48.203   23.395   46.855
H   49.609   22.469   47.065
H   48.206   21.601   46.451
H   47.676   25.378   42.975
H   47.318   23.727   42.989
H   49.894   23.672   42.038
H   49.417   25.275   41.547
H   53.070   17.517   44.800
H   52.270   18.189   43.499
H   51.654   18.311   45.158
H   57.896   18.169   44.790
H   57.113   19.328   46.016
H   57.556   19.798   44.360
H   58.202   23.821   44.953
H   57.746   25.897   46.984
H   59.132   24.166   47.569
H   59.507   22.971   46.314
H   58.272   22.567   47.460
H   59.201   25.935   44.556
H   58.554   27.356   45.482
H   58.233   26.967   42.927
H   57.078   27.928   43.773
H   56.758   26.199   43.454
H   54.226   29.967   46.675
H   53.437   30.653   45.316
H   54.894   29.698   45.012
H   49.001   27.353   45.384
H   46.439   29.281   42.484
H   47.147   28.107   41.372
H   48.001   29.633   41.834
H   47.393   24.893   37.702
H   46.906   23.212   38.127
H   49.380   24.252   36.712
H   48.530   20.190   37.729
H   47.904   20.577   36.101
H   47.234   21.277   37.563
H   50.538   22.866   35.504
H   49.592   21.638   34.959
H   50.802   20.060   36.857
H   51.922   21.353   36.384
H   51.194   20.597   33.855
H   50.673   19.125   34.725
H   53.407   19.842   35.589
H   54.293   20.737   34.082
H   53.017   20.448   32.852
H   54.423   19.235   33.220
H   52.555   17.785   35.971
H   54.010   17.615   34.980
H   51.921   17.802   33.044
H   51.192   16.899   34.353
H   53.222   15.424   34.393
H   53.697   16.108   32.922
H   51.151   15.559   32.091
H   50.992   14.118   34.377
H   51.562   12.858   33.420
H   50.186   13.816   32.863
H   52.261   13.332   31.243
H   53.805   14.065   31.836
H   52.240   15.890   30.079
H   52.373   14.310   29.285
H   54.512   14.555   28.732
H   54.842   14.918   30.424
H   53.587   17.332   29.187
H   55.004   17.601   27.291
H   55.173   15.892   27.299
H   56.407   16.851   28.294
H   54.457   17.755   31.088
H   55.693   18.368   29.979
H   55.970   16.791   30.808
