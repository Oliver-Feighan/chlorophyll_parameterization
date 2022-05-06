%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1201_chromophore_25 TDDFT with cam-b3lyp functional

0 1
Mg   -2.157   34.572   26.150
C   -3.323   32.932   29.007
C   -0.750   36.994   28.172
C   -1.466   36.404   23.364
C   -3.959   32.262   24.204
N   -1.967   34.889   28.385
C   -2.489   34.027   29.336
C   -2.142   34.509   30.753
C   -1.430   35.918   30.487
C   -1.431   35.994   28.904
C   -2.154   37.073   31.167
C   -1.302   33.389   31.642
C   -1.831   33.060   33.036
C   -0.841   33.002   34.247
O   0.247   32.387   34.264
O   -1.303   33.787   35.290
N   -1.196   36.407   25.796
C   -0.633   37.202   26.755
C   0.050   38.338   26.066
C   -0.272   38.264   24.693
C   -1.021   36.993   24.553
C   0.792   39.537   26.830
C   -0.001   39.263   23.577
O   -0.251   39.134   22.393
C   0.705   40.549   23.792
N   -2.670   34.336   24.089
C   -2.179   35.220   23.204
C   -2.504   34.684   21.745
C   -3.325   33.393   21.969
C   -3.273   33.285   23.502
C   -3.376   35.662   20.910
C   -2.761   32.131   21.219
C   -1.349   31.678   21.601
N   -3.327   32.978   26.486
C   -4.039   32.187   25.625
C   -4.788   31.188   26.337
C   -4.550   31.472   27.712
C   -3.673   32.556   27.749
C   -5.797   30.284   25.788
C   -4.827   31.058   29.090
O   -5.604   30.204   29.511
C   -4.017   31.912   29.994
C   -4.943   32.546   31.043
O   -5.536   33.604   30.937
O   -4.914   31.765   32.217
C   -5.815   32.355   33.243
C   -0.453   33.813   36.525
C   -1.323   34.377   37.712
C   -1.776   33.791   38.811
C   -1.690   32.279   39.133
C   -2.294   34.603   39.999
C   -1.230   35.275   40.864
C   -0.987   36.735   40.705
C   -1.306   37.557   41.984
C   -1.783   38.953   41.697
C   0.032   37.696   42.835
C   0.074   36.664   44.028
C   0.993   37.079   45.249
C   2.104   36.030   45.685
C   1.759   35.350   46.984
C   3.601   36.533   45.708
C   4.581   35.506   45.022
C   4.965   35.819   43.519
C   4.607   34.706   42.469
C   5.515   34.942   41.238
C   3.166   35.016   42.096
H   -0.155   37.658   28.803
H   -1.234   36.889   22.414
H   -4.364   31.547   23.485
H   -3.147   34.583   31.170
H   -0.383   35.866   30.784
H   -1.474   37.855   31.506
H   -2.712   36.697   32.025
H   -2.850   37.462   30.424
H   -0.272   33.728   31.758
H   -1.091   32.434   31.160
H   -2.337   32.096   32.985
H   -2.585   33.831   33.190
H   0.805   39.370   27.907
H   0.343   40.528   26.769
H   1.808   39.402   26.458
H   0.096   41.161   24.457
H   0.742   41.201   22.919
H   1.729   40.431   24.148
H   -1.578   34.298   21.319
H   -4.345   33.544   21.617
H   -3.254   36.687   21.258
H   -4.455   35.514   20.969
H   -3.180   35.560   19.843
H   -2.719   32.609   20.240
H   -3.477   31.309   21.230
H   -1.312   30.590   21.659
H   -0.952   32.058   22.542
H   -0.663   31.887   20.779
H   -6.468   30.015   26.604
H   -5.331   29.394   25.367
H   -6.360   30.844   25.041
H   -3.295   31.226   30.437
H   -6.834   32.519   32.891
H   -5.429   33.278   33.677
H   -5.921   31.632   34.052
H   0.391   34.470   36.313
H   -0.062   32.816   36.727
H   -1.398   35.462   37.788
H   -2.659   31.823   39.338
H   -1.032   32.213   40.000
H   -1.261   31.745   38.285
H   -2.896   34.021   40.697
H   -2.848   35.434   39.564
H   -0.306   34.728   40.677
H   -1.538   35.078   41.891
H   -1.471   37.242   39.871
H   0.087   36.813   40.536
H   -2.069   36.996   42.522
H   -1.692   39.271   40.659
H   -1.316   39.801   42.199
H   -2.776   39.060   42.134
H   0.288   38.704   43.162
H   0.926   37.415   42.279
H   0.480   35.724   43.656
H   -0.928   36.533   44.437
H   0.389   37.317   46.124
H   1.537   37.941   44.864
H   1.864   35.225   44.990
H   2.453   34.549   47.237
H   0.745   34.950   47.016
H   1.845   36.088   47.782
H   3.934   36.825   46.704
H   3.596   37.399   45.047
H   4.019   34.572   45.036
H   5.462   35.399   45.654
H   6.044   35.939   43.616
H   4.535   36.784   43.251
H   4.650   33.673   42.815
H   6.564   34.821   41.506
H   5.316   35.908   40.773
H   5.232   34.107   40.598
H   2.580   34.107   41.961
H   3.179   35.585   41.166
H   2.651   35.655   42.813

