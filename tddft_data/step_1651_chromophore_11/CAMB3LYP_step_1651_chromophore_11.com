%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1651_chromophore_11 TDDFT with cam-b3lyp functional

0 1
Mg   52.993   23.531   44.427
C   49.965   25.343   43.985
C   51.203   20.616   43.507
C   55.840   21.854   44.330
C   54.758   26.509   44.793
N   50.803   23.070   43.683
C   49.757   23.951   43.721
C   48.390   23.188   43.492
C   48.801   21.677   43.383
C   50.389   21.746   43.563
C   48.151   20.731   44.438
C   47.546   23.761   42.280
C   48.165   23.874   40.825
C   47.702   22.826   39.819
O   46.962   21.892   39.953
O   48.307   23.098   38.618
N   53.524   21.585   43.863
C   52.629   20.530   43.548
C   53.345   19.284   43.473
C   54.740   19.605   43.753
C   54.830   21.006   43.997
C   52.778   17.987   43.034
C   55.890   18.623   43.664
O   55.663   17.441   43.445
C   57.286   19.069   43.799
N   54.963   24.116   44.392
C   55.957   23.208   44.465
C   57.288   23.915   44.693
C   57.034   25.418   44.685
C   55.480   25.328   44.610
C   58.111   23.568   45.979
C   57.691   26.276   43.547
C   57.011   26.393   42.089
N   52.542   25.573   44.453
C   53.301   26.651   44.755
C   52.494   27.828   44.864
C   51.207   27.348   44.487
C   51.289   26.010   44.233
C   52.932   29.244   45.140
C   49.842   27.865   44.318
O   49.508   29.039   44.348
C   49.016   26.561   43.954
C   48.566   26.830   42.620
O   49.247   26.729   41.580
O   47.225   27.310   42.609
C   46.730   27.738   41.290
C   48.003   22.114   37.522
C   49.193   22.109   36.587
C   49.581   21.043   35.882
C   48.681   19.841   35.531
C   50.874   21.127   34.989
C   51.938   20.088   35.261
C   52.246   19.100   34.079
C   53.685   18.654   33.968
C   54.354   19.075   32.623
C   53.898   17.153   34.244
C   52.956   16.250   33.391
C   53.702   15.187   32.640
C   53.023   14.774   31.332
C   51.799   13.829   31.764
C   53.992   13.988   30.337
C   53.872   14.673   28.989
C   54.198   13.588   27.949
C   55.370   13.999   27.069
C   55.036   13.696   25.630
C   56.776   13.657   27.505
H   50.655   19.673   43.569
H   56.822   21.431   44.549
H   55.279   27.467   44.857
H   47.897   23.388   44.444
H   48.499   21.393   42.375
H   48.996   20.183   44.853
H   47.421   20.049   44.001
H   47.679   21.256   45.269
H   47.319   24.782   42.586
H   46.642   23.153   42.258
H   49.213   23.590   40.917
H   48.021   24.845   40.352
H   51.761   18.088   42.655
H   52.925   17.238   43.811
H   53.303   17.543   42.188
H   57.971   18.256   43.559
H   57.386   19.407   44.831
H   57.371   19.880   43.076
H   57.886   23.768   43.794
H   57.296   25.837   45.657
H   57.423   23.411   46.810
H   58.817   24.371   46.190
H   58.681   22.669   45.744
H   58.706   25.903   43.413
H   57.793   27.273   43.977
H   56.021   25.936   42.109
H   57.603   25.871   41.337
H   56.777   27.436   41.878
H   52.348   29.965   44.568
H   53.985   29.419   44.922
H   52.824   29.499   46.195
H   48.169   26.419   44.625
H   47.537   27.801   40.561
H   46.221   28.696   41.403
H   45.989   27.094   40.817
H   47.170   22.504   36.938
H   47.798   21.079   37.794
H   49.961   22.856   36.784
H   49.346   19.145   35.020
H   47.774   20.254   35.088
H   48.462   19.218   36.397
H   51.311   22.119   35.103
H   50.532   21.042   33.958
H   51.599   19.403   36.038
H   52.821   20.687   35.488
H   51.847   19.519   33.155
H   51.612   18.226   34.228
H   54.253   19.156   34.751
H   53.616   19.600   32.016
H   54.564   18.151   32.084
H   55.249   19.648   32.865
H   53.843   16.914   35.306
H   54.923   16.844   34.040
H   52.349   16.715   32.614
H   52.338   15.604   34.015
H   53.753   14.321   33.300
H   54.720   15.527   32.449
H   52.646   15.712   30.925
H   50.858   14.325   31.522
H   51.822   13.613   32.832
H   51.886   12.871   31.252
H   53.899   12.904   30.401
H   54.977   14.151   30.773
H   54.607   15.476   28.941
H   52.862   15.044   28.814
H   53.259   13.452   27.413
H   54.412   12.621   28.403
H   55.377   15.088   27.040
H   54.181   13.020   25.622
H   55.835   13.281   25.015
H   54.750   14.592   25.078
H   56.741   12.680   27.988
H   57.193   14.427   28.154
H   57.415   13.696   26.623

