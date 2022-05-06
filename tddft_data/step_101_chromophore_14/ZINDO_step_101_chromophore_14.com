%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_101_chromophore_14 ZINDO

0 1
Mg   46.631   44.883   44.835
C   43.185   44.778   44.033
C   46.930   41.482   44.241
C   49.969   45.239   44.892
C   46.241   48.362   45.168
N   45.292   43.405   44.128
C   43.882   43.557   43.832
C   43.274   42.252   43.228
C   44.409   41.200   43.605
C   45.660   42.067   44.092
C   44.096   39.999   44.641
C   43.258   42.443   41.752
C   41.891   42.153   41.091
C   41.760   41.295   39.858
O   41.193   40.171   39.764
O   42.469   41.919   38.831
N   48.234   43.487   44.779
C   48.128   42.159   44.567
C   49.443   41.598   44.356
C   50.411   42.638   44.493
C   49.536   43.889   44.805
C   49.735   40.121   44.084
C   51.890   42.490   44.432
O   52.335   41.367   44.264
C   52.836   43.649   44.578
N   47.920   46.548   44.773
C   49.282   46.447   44.834
C   49.953   47.825   44.925
C   48.768   48.724   45.343
C   47.558   47.862   45.097
C   51.228   47.983   45.871
C   48.745   50.148   44.629
C   48.745   51.405   45.479
N   45.084   46.308   44.724
C   45.005   47.699   44.924
C   43.632   48.188   44.756
C   42.905   47.078   44.367
C   43.804   45.969   44.383
C   43.144   49.686   44.734
C   41.559   46.582   43.983
O   40.410   47.115   43.972
C   41.706   44.993   43.813
C   40.847   44.293   44.769
O   39.851   43.638   44.483
O   41.297   44.400   46.086
C   40.476   43.634   47.039
C   42.295   41.327   37.485
C   42.953   42.234   36.438
C   44.257   42.413   36.068
C   45.432   41.581   36.658
C   44.616   43.430   35.049
C   45.145   43.042   33.684
C   44.710   44.031   32.570
C   45.369   43.710   31.143
C   46.529   44.705   30.807
C   44.293   43.633   30.052
C   44.578   42.532   28.981
C   43.686   42.842   27.833
C   44.364   43.022   26.483
C   43.604   44.071   25.736
C   44.382   41.673   25.642
C   45.830   41.216   25.336
C   45.956   41.005   23.794
C   46.682   42.192   23.190
C   46.248   42.251   21.650
C   48.240   42.117   23.259
H   46.972   40.425   43.969
H   51.041   45.440   44.930
H   46.264   49.422   45.431
H   42.361   42.139   43.813
H   44.669   40.632   42.712
H   44.500   39.062   44.257
H   43.047   39.984   44.937
H   44.723   40.173   45.516
H   44.041   41.864   41.263
H   43.540   43.466   41.502
H   41.443   43.113   40.834
H   41.194   41.525   41.646
H   48.802   39.566   43.987
H   50.316   39.667   44.887
H   50.263   39.971   43.143
H   52.603   43.999   45.584
H   52.610   44.370   43.792
H   53.866   43.295   44.615
H   50.244   48.115   43.916
H   48.792   48.812   46.429
H   52.180   47.953   45.340
H   51.252   47.264   46.690
H   51.298   48.968   46.332
H   47.879   50.273   43.978
H   49.647   50.092   44.020
H   49.532   52.123   45.248
H   48.838   51.104   46.523
H   47.777   51.901   45.412
H   42.453   49.916   43.924
H   44.116   50.177   44.672
H   42.597   49.856   45.662
H   41.318   44.781   42.816
H   40.974   42.672   47.162
H   39.467   43.292   46.809
H   40.405   44.192   47.972
H   41.225   41.299   37.279
H   42.652   40.297   37.508
H   42.183   42.776   35.889
H   46.239   42.207   37.038
H   45.994   40.970   35.953
H   45.045   40.967   37.471
H   45.292   44.212   35.395
H   43.753   44.069   34.864
H   44.898   42.029   33.366
H   46.219   43.157   33.832
H   45.048   44.989   32.965
H   43.624   44.108   32.523
H   45.862   42.740   31.205
H   46.588   45.547   31.497
H   46.411   45.094   29.796
H   47.503   44.224   30.888
H   44.066   44.661   29.767
H   43.410   43.317   30.606
H   44.182   41.607   29.399
H   45.631   42.386   28.740
H   42.996   43.679   27.935
H   43.007   41.992   27.770
H   45.330   43.477   26.699
H   43.829   45.024   26.215
H   42.529   43.889   25.699
H   43.997   44.078   24.720
H   43.801   41.811   24.730
H   43.810   40.943   26.215
H   46.024   40.288   25.875
H   46.601   41.896   25.697
H   44.962   40.956   23.352
H   46.570   40.104   23.774
H   46.339   43.079   23.724
H   46.068   43.318   21.525
H   45.352   41.702   21.360
H   47.067   41.992   20.980
H   48.585   41.330   23.930
H   48.493   43.114   23.620
H   48.781   41.958   22.327
