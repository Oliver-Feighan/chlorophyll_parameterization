%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_901_chromophore_9 TDDFT with cam-b3lyp functional

0 1
Mg   35.538   1.289   29.795
C   33.240   2.435   32.320
C   37.979   1.420   32.226
C   37.812   0.783   27.409
C   32.986   1.852   27.411
N   35.601   1.939   31.931
C   34.541   2.203   32.802
C   34.941   2.280   34.247
C   36.586   2.366   34.137
C   36.735   1.885   32.712
C   37.209   3.732   34.487
C   34.426   1.121   34.986
C   34.434   1.247   36.508
C   35.211   2.351   37.220
O   34.746   3.453   37.722
O   36.549   1.963   37.353
N   37.624   0.834   29.802
C   38.427   1.053   30.906
C   39.833   0.797   30.500
C   39.784   0.539   29.065
C   38.367   0.789   28.688
C   40.941   0.845   31.471
C   40.894   0.309   28.151
O   40.798   0.170   26.941
C   42.294   0.274   28.702
N   35.357   1.146   27.708
C   36.475   0.897   26.976
C   36.126   0.755   25.453
C   34.577   1.107   25.453
C   34.276   1.434   26.957
C   37.074   1.692   24.551
C   33.766   -0.056   24.890
C   33.335   -1.157   25.832
N   33.548   2.081   29.787
C   32.610   2.133   28.725
C   31.315   2.563   29.254
C   31.571   2.662   30.638
C   32.911   2.413   30.938
C   30.075   2.884   28.453
C   30.869   3.008   31.883
O   29.662   3.376   31.992
C   31.914   2.845   33.034
C   31.968   4.159   33.763
O   32.305   5.234   33.237
O   31.793   3.953   35.105
C   31.729   5.232   35.863
C   37.406   2.952   37.978
C   38.104   2.311   39.119
C   37.876   2.416   40.477
C   36.706   2.998   41.189
C   39.051   1.980   41.385
C   39.534   0.557   41.047
C   39.436   -0.434   42.276
C   40.856   -0.914   42.745
C   40.726   -2.330   43.374
C   41.603   0.076   43.702
C   42.754   0.801   42.901
C   43.512   1.824   43.672
C   44.283   2.779   42.777
C   45.844   2.823   43.093
C   43.711   4.237   42.984
C   43.966   5.347   41.863
C   44.702   6.637   42.295
C   43.922   7.914   41.918
C   44.802   9.142   42.134
C   42.657   8.001   42.840
H   38.755   1.347   32.991
H   38.562   0.729   26.617
H   32.186   1.925   26.672
H   34.564   3.212   34.669
H   37.039   1.648   34.820
H   38.048   3.678   35.179
H   36.503   4.506   34.788
H   37.647   4.154   33.582
H   34.966   0.216   34.704
H   33.422   0.793   34.718
H   34.577   0.250   36.924
H   33.396   1.502   36.719
H   41.402   -0.138   31.571
H   40.574   1.186   32.439
H   41.768   1.475   31.144
H   42.739   1.227   28.987
H   43.007   -0.212   28.036
H   42.297   -0.408   29.553
H   36.344   -0.280   25.188
H   34.275   1.945   24.825
H   37.771   1.016   24.055
H   37.660   2.434   25.093
H   36.584   2.199   23.719
H   34.364   -0.506   24.097
H   32.892   0.373   24.400
H   33.736   -1.045   26.840
H   33.606   -2.156   25.491
H   32.266   -1.092   26.032
H   30.331   2.822   27.395
H   29.679   3.871   28.688
H   29.292   2.158   28.673
H   31.628   2.019   33.685
H   31.032   5.940   35.414
H   32.741   5.628   35.950
H   31.368   4.835   36.811
H   37.040   3.954   38.198
H   38.222   3.148   37.282
H   39.027   1.842   38.777
H   36.348   2.142   41.761
H   35.953   3.490   40.573
H   37.012   3.719   41.948
H   38.730   2.022   42.426
H   39.887   2.659   41.218
H   40.508   0.484   40.564
H   38.866   0.217   40.256
H   38.639   -1.147   42.066
H   39.089   0.177   43.110
H   41.513   -1.007   41.880
H   40.576   -3.033   42.554
H   39.823   -2.282   43.982
H   41.584   -2.672   43.952
H   42.083   -0.416   44.549
H   40.935   0.869   44.036
H   42.448   1.143   41.912
H   43.409   -0.051   42.718
H   44.211   1.201   44.229
H   42.914   2.332   44.428
H   44.217   2.505   41.724
H   46.095   2.225   43.969
H   46.214   3.847   43.060
H   46.299   2.244   42.289
H   43.956   4.584   43.987
H   42.630   4.096   42.973
H   42.970   5.647   41.536
H   44.477   4.909   41.006
H   45.556   6.551   41.623
H   44.936   6.576   43.358
H   43.447   7.843   40.940
H   44.114   9.984   42.215
H   45.282   9.174   41.156
H   45.567   9.111   42.910
H   41.864   8.356   42.182
H   42.707   8.729   43.650
H   42.378   6.999   43.167

