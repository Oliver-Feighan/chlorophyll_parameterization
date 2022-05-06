%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_21 TDDFT with cam-b3lyp functional

0 1
Mg   16.033   51.092   25.069
C   17.956   49.760   27.810
C   13.561   51.743   27.202
C   14.388   52.015   22.323
C   18.567   49.603   23.093
N   15.816   50.679   27.211
C   16.716   50.279   28.135
C   16.271   50.680   29.485
C   14.718   50.879   29.396
C   14.692   51.129   27.832
C   13.787   49.800   29.928
C   17.044   52.018   29.933
C   17.464   52.110   31.441
C   16.603   51.391   32.448
O   16.901   50.406   33.140
O   15.370   51.976   32.593
N   14.190   51.865   24.801
C   13.302   52.039   25.785
C   12.102   52.704   25.247
C   12.250   52.735   23.803
C   13.600   52.201   23.592
C   10.973   53.116   26.240
C   11.276   53.192   22.711
O   11.546   53.157   21.518
C   9.952   53.610   23.124
N   16.466   50.881   23.056
C   15.648   51.421   22.100
C   16.158   51.116   20.668
C   17.484   50.291   20.965
C   17.519   50.278   22.455
C   15.251   50.323   19.719
C   18.725   50.933   20.188
C   19.161   50.191   18.889
N   17.851   49.822   25.276
C   18.763   49.507   24.403
C   19.832   48.755   24.966
C   19.606   48.861   26.336
C   18.392   49.515   26.478
C   21.042   48.058   24.243
C   20.121   48.612   27.599
O   21.133   48.102   27.990
C   19.124   49.275   28.599
C   18.716   48.271   29.725
O   17.886   47.367   29.580
O   19.296   48.467   30.987
C   18.733   47.787   32.068
C   14.562   51.381   33.652
C   13.148   51.960   33.744
C   12.474   52.069   34.898
C   12.853   51.380   36.232
C   11.190   52.742   34.965
C   11.118   53.884   36.005
C   10.353   55.190   35.601
C   8.829   55.239   35.815
C   8.014   54.870   34.503
C   8.389   56.550   36.451
C   7.041   56.411   37.020
C   5.872   56.871   36.033
C   4.925   57.815   36.727
C   4.059   57.121   37.870
C   3.948   58.657   35.825
C   4.520   60.000   35.455
C   3.538   61.055   35.552
C   3.827   61.961   36.790
C   2.494   62.705   37.208
C   4.966   62.983   36.404
H   12.788   51.977   27.938
H   13.954   52.202   21.338
H   19.297   49.177   22.402
H   16.478   49.839   30.148
H   14.499   51.871   29.793
H   13.081   49.506   29.152
H   13.190   50.082   30.796
H   14.321   48.857   30.048
H   16.601   52.923   29.517
H   17.996   51.917   29.411
H   17.507   53.184   31.623
H   18.442   51.747   31.756
H   11.499   53.338   27.169
H   10.180   52.373   26.329
H   10.533   54.046   25.880
H   9.940   54.555   23.666
H   9.538   52.789   23.710
H   9.305   53.567   22.248
H   16.317   52.085   20.196
H   17.327   49.263   20.638
H   14.233   50.196   20.086
H   15.625   49.351   19.396
H   15.110   50.984   18.863
H   19.507   51.253   20.876
H   18.401   51.881   19.757
H   18.595   50.514   18.015
H   18.999   49.142   19.138
H   20.195   50.290   18.559
H   21.608   47.522   25.006
H   21.497   48.939   23.790
H   20.624   47.376   23.503
H   19.692   50.048   29.117
H   19.403   47.013   32.443
H   17.755   47.311   32.007
H   18.658   48.494   32.895
H   15.075   51.421   34.613
H   14.292   50.342   33.459
H   12.779   52.524   32.887
H   11.999   50.998   36.791
H   13.434   52.096   36.813
H   13.443   50.463   36.221
H   10.485   51.941   35.185
H   10.911   53.097   33.973
H   12.083   54.283   36.316
H   10.550   53.547   36.873
H   10.586   55.285   34.540
H   10.828   55.952   36.220
H   8.551   54.422   36.482
H   8.476   54.081   33.909
H   7.794   55.694   33.824
H   7.052   54.493   34.849
H   8.442   57.371   35.736
H   9.018   56.829   37.296
H   6.985   56.952   37.964
H   6.922   55.343   37.206
H   5.221   56.054   35.721
H   6.308   57.326   35.144
H   5.436   58.586   37.303
H   4.154   57.658   38.814
H   4.375   56.096   38.069
H   3.008   57.137   37.583
H   2.989   58.714   36.339
H   3.764   58.114   34.898
H   5.023   59.977   34.488
H   5.409   60.284   36.017
H   2.507   60.700   35.571
H   3.647   61.651   34.645
H   4.132   61.428   37.690
H   2.385   63.742   36.893
H   2.448   62.666   38.296
H   1.620   62.166   36.842
H   5.570   63.429   37.194
H   4.561   63.919   36.021
H   5.572   62.619   35.574

