%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1501_chromophore_22 TDDFT with PBE1PBE functional

0 1
Mg   8.781   47.858   25.220
C   6.994   48.021   28.145
C   11.273   49.654   26.849
C   10.446   48.019   22.207
C   5.901   46.803   23.428
N   9.011   48.803   27.177
C   8.264   48.634   28.285
C   8.838   49.270   29.598
C   9.902   50.186   28.958
C   10.070   49.552   27.542
C   9.443   51.672   28.887
C   9.357   48.355   30.644
C   9.111   48.879   32.069
C   8.656   47.777   33.167
O   7.897   46.837   32.904
O   9.299   47.949   34.399
N   10.714   48.541   24.740
C   11.552   49.204   25.568
C   12.767   49.370   24.821
C   12.501   49.081   23.471
C   11.153   48.499   23.426
C   14.135   49.893   25.325
C   13.433   49.136   22.258
O   13.099   48.796   21.128
C   14.880   49.566   22.466
N   8.288   47.364   23.136
C   9.094   47.523   22.124
C   8.430   47.002   20.821
C   7.080   46.431   21.236
C   7.078   46.813   22.727
C   8.442   48.027   19.600
C   6.735   44.929   21.034
C   5.362   44.639   20.542
N   6.872   47.366   25.634
C   5.811   47.038   24.818
C   4.608   47.133   25.599
C   5.059   47.299   26.912
C   6.424   47.487   26.911
C   3.219   46.967   25.194
C   4.658   47.455   28.275
O   3.467   47.388   28.655
C   5.913   47.744   29.115
C   6.159   46.575   29.955
O   7.203   45.990   30.018
O   4.987   46.346   30.662
C   5.018   45.119   31.516
C   8.811   47.179   35.553
C   9.556   47.723   36.749
C   9.112   47.889   38.059
C   7.755   47.466   38.648
C   9.932   48.659   39.083
C   10.870   47.861   40.021
C   11.149   48.500   41.367
C   10.927   47.617   42.677
C   10.519   48.410   43.946
C   12.182   46.722   43.051
C   11.779   45.262   43.350
C   12.157   44.716   44.745
C   13.535   44.024   44.703
C   14.296   44.475   45.957
C   13.482   42.498   44.547
C   14.803   41.911   43.867
C   14.571   41.421   42.423
C   15.743   41.797   41.471
C   15.220   41.585   40.040
C   16.947   40.833   41.777
H   11.943   50.436   27.212
H   11.028   48.049   21.283
H   4.945   46.428   23.056
H   8.088   49.842   30.143
H   10.877   50.150   29.444
H   8.363   51.811   28.941
H   9.723   52.206   27.979
H   9.881   52.246   29.703
H   10.430   48.190   30.550
H   8.855   47.388   30.618
H   8.501   49.780   32.137
H   10.114   49.203   32.347
H   15.009   49.274   25.123
H   14.132   50.049   26.404
H   14.329   50.867   24.874
H   15.035   50.510   22.987
H   15.312   49.698   21.474
H   15.418   48.809   23.036
H   8.801   46.075   20.384
H   6.340   47.101   20.798
H   9.143   48.845   19.770
H   7.431   48.432   19.561
H   8.785   47.485   18.718
H   6.912   44.245   21.864
H   7.365   44.583   20.215
H   4.820   44.028   21.264
H   5.361   44.105   19.591
H   4.798   45.564   20.422
H   2.748   47.804   24.679
H   2.779   46.831   26.182
H   3.008   46.048   24.647
H   5.672   48.591   29.758
H   5.770   44.338   31.398
H   4.034   44.677   31.359
H   5.102   45.494   32.536
H   9.115   46.140   35.425
H   7.773   47.400   35.804
H   10.437   48.323   36.521
H   7.813   46.831   39.533
H   7.178   46.861   37.949
H   7.150   48.310   38.979
H   9.239   49.355   39.554
H   10.598   49.259   38.463
H   11.830   47.653   39.548
H   10.409   46.895   40.228
H   10.503   49.364   41.526
H   12.207   48.760   41.332
H   10.096   46.942   42.471
H   11.098   49.326   44.062
H   10.690   47.833   44.855
H   9.449   48.579   43.821
H   12.575   47.186   43.955
H   13.071   46.537   42.448
H   12.008   44.611   42.506
H   10.689   45.246   43.378
H   11.404   43.964   44.980
H   12.148   45.508   45.494
H   14.104   44.427   43.865
H   13.816   44.024   46.826
H   14.344   45.560   46.048
H   15.307   44.108   45.781
H   12.607   42.258   43.943
H   13.476   42.146   45.578
H   15.124   41.152   44.581
H   15.608   42.642   43.940
H   13.647   41.829   42.011
H   14.480   40.335   42.458
H   15.998   42.854   41.543
H   14.143   41.552   39.873
H   15.319   40.530   39.786
H   15.724   42.148   39.254
H   17.761   41.528   41.983
H   17.302   40.188   40.973
H   16.951   40.284   42.719
