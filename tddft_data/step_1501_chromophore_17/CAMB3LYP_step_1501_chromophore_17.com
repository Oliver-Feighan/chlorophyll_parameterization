%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1501_chromophore_17 TDDFT with cam-b3lyp functional

0 1
Mg   29.248   59.747   41.489
C   26.091   58.809   40.378
C   30.441   56.789   40.137
C   32.269   61.002   41.686
C   27.903   62.792   42.516
N   28.433   58.154   40.224
C   27.030   57.911   39.955
C   26.858   56.533   39.239
C   28.166   55.856   39.683
C   29.129   57.025   40.018
C   27.945   54.864   40.887
C   26.538   56.654   37.656
C   25.302   55.878   37.132
C   25.030   55.827   35.620
O   23.899   55.775   35.218
O   26.198   55.756   34.852
N   31.123   58.946   41.071
C   31.452   57.728   40.484
C   32.825   57.601   40.510
C   33.402   58.854   40.890
C   32.249   59.706   41.172
C   33.476   56.313   40.012
C   34.907   59.129   40.863
O   35.625   58.207   40.640
C   35.466   60.569   41.095
N   29.954   61.682   42.033
C   31.298   61.927   42.028
C   31.566   63.393   42.628
C   30.129   63.821   43.030
C   29.258   62.747   42.434
C   32.704   63.547   43.675
C   29.812   65.338   42.774
C   29.172   65.623   41.396
N   27.362   60.649   41.658
C   26.973   61.880   42.089
C   25.483   61.917   42.050
C   25.102   60.769   41.317
C   26.291   60.095   41.060
C   24.653   62.978   42.430
C   24.021   59.973   40.898
O   22.815   60.117   40.973
C   24.648   58.716   40.187
C   24.048   57.480   40.730
O   24.365   56.972   41.782
O   23.011   56.980   39.953
C   22.468   55.711   40.340
C   25.946   55.699   33.421
C   27.159   55.934   32.597
C   27.146   56.240   31.333
C   25.910   56.376   30.515
C   28.389   56.334   30.452
C   29.503   55.302   30.750
C   30.729   55.813   31.635
C   31.985   56.178   30.829
C   32.421   57.679   31.174
C   33.242   55.225   30.884
C   33.957   55.124   29.506
C   35.388   55.357   29.726
C   36.361   54.218   29.405
C   36.596   53.474   30.728
C   37.721   54.663   28.764
C   37.787   54.304   27.275
C   37.454   55.543   26.444
C   36.260   55.166   25.454
C   36.888   54.389   24.265
C   35.586   56.423   24.922
H   30.775   55.808   39.794
H   33.245   61.453   41.879
H   27.487   63.753   42.828
H   25.941   56.032   39.549
H   28.582   55.282   38.855
H   28.673   54.065   40.747
H   26.933   54.460   40.912
H   28.230   55.366   41.811
H   27.396   56.348   37.057
H   26.497   57.721   37.437
H   24.418   56.313   37.599
H   25.178   54.868   37.523
H   33.911   56.547   39.041
H   32.802   55.457   39.994
H   34.274   56.062   40.711
H   34.830   61.143   40.420
H   36.542   60.605   40.924
H   35.337   60.737   42.164
H   31.833   63.945   41.727
H   30.075   63.800   44.118
H   33.675   63.400   43.202
H   32.675   62.875   44.533
H   32.856   64.584   43.973
H   30.694   65.953   42.956
H   29.113   65.489   43.597
H   28.106   65.770   41.571
H   29.251   64.831   40.652
H   29.520   66.607   41.081
H   23.691   62.772   41.961
H   25.136   63.883   42.061
H   24.370   63.019   43.482
H   24.474   58.761   39.111
H   23.258   55.047   39.989
H   21.635   55.417   39.701
H   22.278   55.500   41.392
H   25.200   56.436   33.124
H   25.621   54.696   33.142
H   28.142   56.015   33.059
H   26.093   57.282   29.938
H   24.966   56.526   31.039
H   25.677   55.404   30.080
H   28.758   57.358   30.504
H   28.152   56.203   29.396
H   29.878   54.785   29.867
H   29.033   54.466   31.267
H   31.035   55.049   32.350
H   30.260   56.658   32.139
H   31.646   56.333   29.804
H   31.937   58.005   32.095
H   32.058   58.317   30.368
H   33.496   57.780   31.318
H   32.897   54.227   31.156
H   33.869   55.457   31.744
H   33.502   55.806   28.789
H   33.837   54.089   29.186
H   35.581   55.851   30.679
H   35.596   56.182   29.045
H   35.856   53.639   28.632
H   36.998   52.471   30.585
H   35.685   53.267   31.289
H   37.307   54.089   31.280
H   38.541   54.207   29.318
H   37.871   55.711   29.024
H   37.123   53.481   27.009
H   38.827   54.078   27.041
H   38.271   55.882   25.806
H   37.098   56.431   26.965
H   35.529   54.481   25.885
H   37.953   54.600   24.162
H   36.520   54.636   23.269
H   36.587   53.351   24.407
H   35.265   56.199   23.904
H   36.428   57.108   24.829
H   34.830   56.799   25.611

