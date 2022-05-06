%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1451_chromophore_17 TDDFT with blyp functional

0 1
Mg   29.992   59.679   41.359
C   27.034   58.160   40.012
C   31.741   56.931   40.168
C   32.751   61.503   41.627
C   28.019   62.561   41.901
N   29.414   57.741   40.161
C   28.154   57.333   39.834
C   28.168   55.918   39.287
C   29.633   55.435   39.708
C   30.334   56.789   40.077
C   29.839   54.270   40.658
C   27.803   55.923   37.726
C   26.726   54.799   37.286
C   26.047   55.112   35.942
O   24.864   54.924   35.722
O   26.974   55.481   34.908
N   32.005   59.233   40.998
C   32.581   58.054   40.500
C   34.025   58.212   40.508
C   34.284   59.511   40.990
C   33.009   60.188   41.174
C   34.985   57.127   40.092
C   35.666   60.005   41.305
O   36.603   59.218   41.227
C   36.032   61.474   41.740
N   30.278   61.641   41.853
C   31.518   62.118   42.024
C   31.520   63.674   42.338
C   30.011   63.951   42.513
C   29.409   62.616   42.109
C   32.517   64.235   43.458
C   29.487   65.247   41.780
C   29.070   65.000   40.274
N   27.892   60.295   41.160
C   27.230   61.412   41.505
C   25.825   61.320   41.192
C   25.727   60.047   40.540
C   26.998   59.469   40.575
C   24.802   62.320   41.565
C   24.860   59.091   39.837
O   23.677   59.213   39.500
C   25.583   57.752   39.607
C   24.972   56.687   40.500
O   24.986   56.773   41.720
O   24.413   55.759   39.692
C   23.795   54.579   40.352
C   26.346   55.632   33.566
C   27.419   56.236   32.591
C   27.240   56.734   31.365
C   25.923   56.775   30.635
C   28.370   57.297   30.550
C   29.259   56.244   29.933
C   30.730   56.473   30.132
C   31.526   56.010   28.886
C   31.455   57.151   27.758
C   32.983   55.843   29.332
C   33.335   54.383   29.601
C   34.751   54.259   30.141
C   35.762   53.436   29.237
C   36.602   52.610   30.268
C   36.605   54.317   28.244
C   36.234   54.090   26.654
C   35.687   55.351   25.937
C   36.375   55.845   24.643
C   36.401   57.362   24.517
C   35.624   55.154   23.516
H   32.358   56.058   39.945
H   33.654   62.115   41.677
H   27.446   63.469   42.102
H   27.432   55.252   39.736
H   30.195   55.092   38.838
H   28.850   53.970   41.005
H   30.507   54.361   41.515
H   30.227   53.479   40.016
H   28.706   55.668   37.172
H   27.393   56.895   37.451
H   25.905   54.757   38.002
H   27.293   53.869   37.234
H   35.370   57.462   39.129
H   34.428   56.201   39.952
H   35.645   56.781   40.888
H   35.877   62.130   40.884
H   37.101   61.363   41.918
H   35.470   61.847   42.596
H   31.858   64.299   41.512
H   29.872   63.999   43.593
H   33.548   64.217   43.103
H   32.367   63.565   44.304
H   32.195   65.244   43.714
H   30.316   65.954   41.809
H   28.680   65.613   42.415
H   29.466   65.832   39.691
H   28.033   64.819   39.991
H   29.708   64.218   39.862
H   23.918   61.784   41.913
H   24.330   62.908   40.777
H   25.132   63.066   42.288
H   25.634   57.503   38.547
H   24.396   53.680   40.491
H   22.934   54.237   39.778
H   23.337   54.801   41.316
H   25.677   56.486   33.674
H   25.856   54.797   33.065
H   28.458   56.215   32.921
H   26.053   56.192   29.722
H   25.471   57.735   30.388
H   25.184   56.266   31.254
H   28.963   57.997   31.138
H   27.995   57.923   29.741
H   28.945   56.248   28.889
H   29.099   55.303   30.461
H   31.097   56.051   31.068
H   30.993   57.512   30.329
H   31.057   55.152   28.405
H   30.469   57.613   27.807
H   31.578   56.838   26.721
H   32.266   57.834   28.011
H   33.068   56.286   30.324
H   33.651   56.241   28.569
H   33.338   53.844   28.654
H   32.632   53.951   30.313
H   34.672   53.750   31.102
H   35.164   55.256   30.293
H   35.194   52.715   28.649
H   35.953   52.209   31.047
H   37.375   53.241   30.707
H   36.936   51.774   29.654
H   37.634   54.045   28.479
H   36.482   55.349   28.572
H   35.436   53.350   26.615
H   37.161   53.701   26.233
H   35.682   56.281   26.505
H   34.617   55.188   25.804
H   37.401   55.492   24.535
H   36.797   57.794   25.437
H   35.408   57.805   24.450
H   37.103   57.689   23.750
H   35.595   55.765   22.614
H   34.547   55.045   23.642
H   36.046   54.201   23.196

