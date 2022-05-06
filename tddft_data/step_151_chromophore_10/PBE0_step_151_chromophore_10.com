%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_151_chromophore_10 TDDFT with PBE1PBE functional

0 1
Mg   41.520   8.296   28.888
C   43.471   9.572   31.601
C   39.298   7.029   31.329
C   39.660   6.965   26.492
C   44.117   8.979   26.814
N   41.543   8.227   31.331
C   42.374   8.947   32.119
C   41.850   8.804   33.560
C   40.553   7.825   33.489
C   40.441   7.672   31.942
C   40.697   6.542   34.374
C   41.384   10.138   34.270
C   40.544   10.143   35.621
C   40.787   11.288   36.627
O   40.256   12.398   36.648
O   41.702   10.827   37.571
N   39.714   7.327   28.872
C   38.937   6.844   29.969
C   37.665   6.291   29.485
C   37.782   6.387   28.110
C   39.094   6.880   27.745
C   36.610   5.663   30.377
C   36.800   5.946   27.037
O   36.951   5.883   25.813
C   35.413   5.487   27.495
N   41.967   7.783   26.967
C   40.973   7.366   26.095
C   41.476   7.432   24.607
C   42.872   8.102   24.763
C   43.079   8.261   26.271
C   41.547   6.035   23.808
C   42.998   9.360   23.893
C   42.104   10.480   24.484
N   43.370   9.133   29.108
C   44.328   9.372   28.128
C   45.450   10.091   28.661
C   45.118   10.190   30.075
C   43.841   9.631   30.247
C   46.728   10.564   27.958
C   45.646   10.656   31.393
O   46.707   11.164   31.749
C   44.453   10.413   32.373
C   45.001   9.648   33.603
O   45.503   8.473   33.628
O   44.826   10.357   34.710
C   45.188   9.736   36.000
C   42.199   11.812   38.484
C   41.238   12.007   39.735
C   41.471   12.684   40.883
C   42.799   13.164   41.251
C   40.362   12.610   41.969
C   40.137   11.178   42.668
C   38.746   10.589   42.251
C   37.674   10.649   43.433
C   37.501   9.275   44.038
C   36.357   11.194   42.916
C   35.339   11.336   44.175
C   34.341   12.463   44.293
C   33.650   12.433   45.642
C   32.234   13.111   45.629
C   34.550   13.200   46.703
C   34.547   12.353   47.994
C   35.959   12.156   48.580
C   36.000   11.821   50.085
C   37.442   12.152   50.568
C   35.595   10.393   50.518
H   38.613   6.717   32.120
H   39.034   6.697   25.638
H   44.854   9.384   26.118
H   42.687   8.471   34.173
H   39.672   8.399   33.776
H   40.525   5.642   33.783
H   39.932   6.471   35.147
H   41.724   6.382   34.701
H   40.859   10.835   33.618
H   42.333   10.650   34.435
H   40.630   9.218   36.190
H   39.481   10.314   35.451
H   35.681   6.165   30.104
H   36.808   5.838   31.434
H   36.543   4.602   30.139
H   34.854   6.381   27.771
H   35.369   4.865   28.389
H   34.922   4.961   26.677
H   40.859   8.179   24.106
H   43.725   7.466   24.528
H   40.570   5.794   23.389
H   41.743   5.169   24.441
H   42.256   6.060   22.980
H   42.596   9.111   22.910
H   44.047   9.653   23.864
H   41.696   10.230   25.464
H   41.304   10.647   23.763
H   42.612   11.441   24.559
H   46.527   10.792   26.911
H   47.429   9.738   28.086
H   47.154   11.431   28.462
H   44.107   11.421   32.602
H   46.228   9.417   35.930
H   44.404   9.012   36.225
H   45.174   10.563   36.709
H   42.479   12.778   38.065
H   43.073   11.352   38.945
H   40.215   11.678   39.552
H   42.826   13.121   42.340
H   42.700   14.245   41.157
H   43.727   12.939   40.725
H   39.486   12.886   41.383
H   40.445   13.327   42.787
H   40.118   11.315   43.749
H   40.911   10.502   42.306
H   38.898   9.552   41.952
H   38.315   11.146   41.420
H   38.081   11.377   44.136
H   37.874   8.414   43.484
H   36.470   9.042   44.304
H   37.972   9.251   45.020
H   35.893   10.531   42.186
H   36.433   12.112   42.333
H   35.945   11.411   45.077
H   34.739   10.428   44.249
H   33.666   12.238   43.467
H   34.644   13.493   44.105
H   33.495   11.394   45.930
H   32.019   13.511   46.620
H   31.528   12.344   45.311
H   32.227   13.735   44.736
H   34.118   14.175   46.925
H   35.585   13.372   46.409
H   33.949   11.443   48.023
H   34.105   13.068   48.688
H   36.383   13.137   48.362
H   36.481   11.348   48.067
H   35.331   12.479   50.638
H   37.824   11.346   51.196
H   37.424   13.100   51.105
H   38.125   12.238   49.723
H   34.721   10.688   51.098
H   36.259   9.957   51.264
H   35.466   9.796   49.615

