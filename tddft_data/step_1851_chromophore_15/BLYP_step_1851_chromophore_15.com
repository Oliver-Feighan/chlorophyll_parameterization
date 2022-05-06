%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1851_chromophore_15 TDDFT with blyp functional

0 1
Mg   47.032   35.412   28.481
C   45.430   33.191   30.672
C   46.867   37.823   30.924
C   47.789   37.636   26.110
C   46.436   32.859   26.021
N   46.187   35.576   30.489
C   45.640   34.503   31.215
C   45.758   34.732   32.666
C   45.721   36.310   32.650
C   46.335   36.629   31.274
C   44.251   36.967   32.725
C   47.103   34.098   33.225
C   47.111   33.898   34.756
C   46.195   34.836   35.552
O   44.993   34.599   35.834
O   46.848   35.966   36.027
N   47.357   37.403   28.568
C   47.379   38.215   29.646
C   47.960   39.523   29.337
C   48.123   39.519   27.896
C   47.780   38.131   27.446
C   48.283   40.575   30.358
C   48.592   40.673   27.031
O   48.576   40.592   25.785
C   49.115   42.006   27.596
N   47.084   35.313   26.342
C   47.512   36.331   25.585
C   47.448   35.946   24.112
C   47.203   34.428   24.139
C   46.906   34.154   25.588
C   46.426   36.697   23.263
C   48.565   33.582   23.797
C   48.388   32.365   22.847
N   46.441   33.340   28.369
C   46.252   32.511   27.302
C   45.806   31.155   27.847
C   45.451   31.404   29.174
C   45.826   32.742   29.430
C   45.554   29.893   27.065
C   44.933   30.799   30.388
O   44.737   29.641   30.682
C   44.564   32.008   31.211
C   44.701   31.602   32.543
O   45.739   31.218   33.073
O   43.465   31.673   33.160
C   43.511   31.340   34.597
C   46.032   36.821   36.973
C   46.923   37.675   37.910
C   46.575   38.836   38.554
C   45.098   39.314   38.594
C   47.570   39.722   39.187
C   48.594   40.346   38.189
C   49.099   41.741   38.656
C   48.465   43.045   38.068
C   48.952   43.534   36.714
C   48.729   44.179   39.001
C   47.436   44.835   39.298
C   47.514   46.261   39.923
C   46.559   47.178   39.179
C   46.966   48.685   39.647
C   45.101   46.839   39.537
C   44.070   46.897   38.360
C   42.654   47.149   38.850
C   41.633   46.059   38.482
C   41.084   46.193   37.051
C   40.434   46.096   39.533
H   47.022   38.413   31.829
H   48.244   38.246   25.327
H   46.361   32.159   25.186
H   44.897   34.309   33.184
H   46.373   36.662   33.450
H   44.243   37.583   33.624
H   43.550   36.212   33.084
H   43.903   37.499   31.840
H   47.835   34.852   32.933
H   47.262   33.145   32.719
H   48.132   33.964   35.132
H   46.830   32.871   34.994
H   47.642   41.370   29.976
H   49.321   40.895   30.267
H   48.154   40.377   31.422
H   49.212   42.664   26.732
H   50.115   41.865   28.006
H   48.483   42.469   28.354
H   48.424   35.995   23.630
H   46.405   34.145   23.452
H   45.518   36.857   23.845
H   46.387   36.139   22.327
H   46.843   37.640   22.908
H   49.057   33.144   24.665
H   49.345   34.168   23.311
H   48.992   32.611   21.973
H   47.322   32.210   22.684
H   48.806   31.496   23.356
H   45.431   30.069   25.996
H   44.649   29.463   27.494
H   46.288   29.118   27.284
H   43.522   32.217   30.966
H   43.724   32.203   35.227
H   44.223   30.549   34.835
H   42.471   31.128   34.843
H   45.331   36.346   37.660
H   45.389   37.474   36.383
H   47.980   37.439   37.781
H   45.123   40.313   38.160
H   44.697   39.258   39.606
H   44.393   38.732   38.000
H   48.162   39.249   39.971
H   47.031   40.473   39.766
H   48.145   40.473   37.204
H   49.486   39.738   38.038
H   50.110   41.660   38.256
H   49.201   41.748   39.741
H   47.424   42.739   37.965
H   48.054   43.877   36.201
H   49.534   42.780   36.183
H   49.658   44.363   36.772
H   49.536   44.859   38.729
H   49.097   43.850   39.973
H   46.988   44.179   40.044
H   46.763   44.818   38.441
H   48.534   46.606   39.754
H   47.270   46.252   40.985
H   46.652   47.106   38.096
H   46.807   49.379   38.821
H   48.016   48.890   39.855
H   46.354   49.067   40.464
H   44.755   47.592   40.246
H   44.903   45.826   39.886
H   44.222   45.923   37.896
H   44.309   47.751   37.726
H   42.304   48.088   38.422
H   42.623   47.157   39.939
H   42.229   45.152   38.576
H   41.308   47.169   36.622
H   40.008   46.028   37.001
H   41.505   45.423   36.403
H   40.524   45.181   40.118
H   39.433   46.138   39.104
H   40.533   46.938   40.219

