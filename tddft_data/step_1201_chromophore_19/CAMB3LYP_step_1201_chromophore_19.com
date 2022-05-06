%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1201_chromophore_19 TDDFT with cam-b3lyp functional

0 1
Mg   25.989   51.141   26.918
C   23.886   52.035   29.587
C   28.170   49.977   29.329
C   28.091   50.530   24.488
C   23.728   52.596   24.667
N   26.066   51.028   29.272
C   25.040   51.468   30.116
C   25.412   51.150   31.610
C   26.971   50.897   31.492
C   27.066   50.506   29.981
C   27.898   52.151   31.898
C   24.665   49.913   32.206
C   23.857   50.168   33.486
C   24.613   50.918   34.663
O   24.925   52.131   34.613
O   24.773   50.109   35.736
N   27.894   50.396   26.912
C   28.572   49.831   28.014
C   29.857   49.251   27.492
C   29.905   49.640   26.128
C   28.504   50.134   25.786
C   30.952   48.749   28.405
C   31.065   49.379   25.167
O   31.089   49.826   23.981
C   32.372   48.786   25.702
N   25.959   51.605   24.794
C   26.972   51.270   24.062
C   26.777   51.630   22.580
C   25.269   52.313   22.616
C   24.922   52.177   24.082
C   27.908   52.663   22.106
C   24.227   51.727   21.565
C   23.504   50.441   21.926
N   24.275   52.224   27.097
C   23.441   52.651   26.041
C   22.192   53.226   26.622
C   22.337   53.008   28.012
C   23.614   52.366   28.231
C   21.015   53.816   25.838
C   21.654   53.031   29.285
O   20.547   53.509   29.510
C   22.678   52.509   30.386
C   22.901   53.688   31.264
O   23.555   54.672   30.990
O   22.230   53.476   32.448
C   22.560   54.434   33.484
C   25.449   50.767   36.838
C   25.371   49.809   38.069
C   25.879   49.978   39.311
C   26.613   51.266   39.661
C   25.473   48.958   40.414
C   26.587   48.048   41.055
C   26.190   46.568   40.972
C   26.535   45.740   42.159
C   26.637   44.291   41.608
C   25.546   45.833   43.396
C   26.227   45.598   44.769
C   25.601   46.378   45.917
C   26.418   47.613   46.358
C   26.060   48.100   47.817
C   26.392   48.760   45.347
C   27.758   49.399   45.123
C   27.644   50.911   45.373
C   27.111   51.692   44.135
C   28.258   51.853   43.174
C   26.614   53.091   44.565
H   28.879   49.468   29.985
H   28.743   50.306   23.641
H   23.089   52.977   23.867
H   25.292   52.080   32.165
H   27.195   50.018   32.095
H   27.494   53.041   32.382
H   28.221   52.486   30.912
H   28.736   51.787   32.491
H   25.302   49.032   32.284
H   23.921   49.774   31.421
H   23.440   49.239   33.876
H   23.021   50.777   33.143
H   31.310   47.753   28.145
H   30.586   48.638   29.425
H   31.627   49.602   28.478
H   33.052   48.713   24.853
H   32.253   47.820   26.192
H   32.727   49.575   26.365
H   26.782   50.683   22.042
H   25.328   53.385   22.432
H   28.504   52.343   21.251
H   28.683   52.771   22.865
H   27.435   53.621   21.893
H   24.742   51.539   20.623
H   23.570   52.570   21.353
H   23.677   49.849   21.027
H   22.424   50.582   21.918
H   23.855   49.941   22.828
H   21.410   54.666   25.282
H   20.354   54.263   26.581
H   20.523   53.013   25.290
H   22.102   51.706   30.846
H   22.423   55.480   33.210
H   23.635   54.324   33.629
H   22.071   54.161   34.419
H   25.118   51.772   37.102
H   26.510   50.877   36.610
H   24.788   48.908   37.877
H   27.629   51.021   39.971
H   26.119   51.658   40.550
H   26.662   52.054   38.909
H   24.568   48.447   40.085
H   25.065   49.629   41.169
H   26.630   48.265   42.122
H   27.538   48.251   40.562
H   26.861   46.354   40.140
H   25.167   46.364   40.654
H   27.563   45.955   42.449
H   27.692   44.071   41.441
H   26.287   44.235   40.577
H   26.246   43.520   42.271
H   24.879   44.985   43.244
H   24.875   46.692   43.370
H   27.303   45.771   44.733
H   26.131   44.581   45.149
H   25.463   45.706   46.764
H   24.618   46.681   45.557
H   27.414   47.203   46.522
H   25.098   47.655   48.070
H   26.014   49.183   47.925
H   26.735   47.867   48.641
H   25.598   49.461   45.605
H   26.111   48.479   44.333
H   28.186   49.253   44.131
H   28.510   49.045   45.829
H   28.651   51.257   45.609
H   27.053   51.241   46.227
H   26.278   51.138   43.701
H   28.123   52.733   42.545
H   28.352   51.056   42.436
H   29.240   51.880   43.647
H   26.235   53.085   45.587
H   25.764   53.482   44.006
H   27.318   53.923   44.563

