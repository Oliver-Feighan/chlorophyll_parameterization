%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1501_chromophore_1 TDDFT with wB97XD functional

0 1
Mg   -0.759   17.348   26.580
C   -1.232   14.961   29.287
C   -1.315   19.879   28.940
C   -0.712   19.553   24.144
C   -0.476   14.738   24.525
N   -1.379   17.362   28.936
C   -1.472   16.293   29.804
C   -1.671   16.854   31.227
C   -1.644   18.374   31.022
C   -1.389   18.569   29.542
C   -2.938   19.196   31.444
C   -0.784   16.202   32.418
C   -0.451   17.172   33.582
C   -0.484   16.629   34.976
O   -0.597   15.413   35.250
O   -0.428   17.640   35.919
N   -0.982   19.435   26.531
C   -1.046   20.326   27.631
C   -1.041   21.656   27.070
C   -0.800   21.603   25.674
C   -0.753   20.184   25.401
C   -1.328   22.905   27.909
C   -0.625   22.620   24.678
O   -0.547   22.463   23.462
C   -0.501   24.128   25.106
N   -0.672   17.169   24.602
C   -0.628   18.199   23.741
C   -0.764   17.735   22.350
C   -0.209   16.286   22.508
C   -0.430   16.045   23.988
C   -2.247   17.768   21.801
C   1.351   16.072   22.131
C   2.510   16.496   23.100
N   -0.918   15.262   26.766
C   -0.784   14.310   25.876
C   -0.955   13.035   26.429
C   -1.046   13.261   27.737
C   -1.052   14.592   27.968
C   -1.068   11.822   25.580
C   -1.181   12.500   29.010
O   -1.326   11.291   29.175
C   -1.284   13.628   30.096
C   -0.147   13.320   30.978
O   0.971   13.760   30.896
O   -0.573   12.361   31.842
C   0.294   12.072   32.979
C   -0.721   17.105   37.239
C   -0.247   18.131   38.244
C   0.046   18.030   39.575
C   -0.244   16.719   40.356
C   0.385   19.232   40.400
C   1.766   19.325   41.089
C   2.685   20.506   40.614
C   2.730   21.639   41.730
C   1.773   22.810   41.344
C   4.172   22.032   42.060
C   4.258   23.048   43.259
C   5.050   22.546   44.454
C   4.325   22.893   45.780
C   5.441   22.873   46.828
C   3.118   22.043   46.209
C   1.931   22.758   46.914
C   0.659   23.113   46.070
C   -0.323   21.974   45.897
C   -0.843   21.879   44.408
C   -1.585   22.295   46.701
H   -1.274   20.640   29.723
H   -0.625   20.194   23.264
H   -0.432   13.976   23.744
H   -2.661   16.491   31.506
H   -0.824   18.818   31.587
H   -3.452   19.639   30.591
H   -2.689   19.786   32.326
H   -3.705   18.461   31.688
H   0.170   16.026   31.920
H   -1.142   15.253   32.817
H   -1.174   17.987   33.563
H   0.551   17.567   33.415
H   -1.868   23.635   27.306
H   -0.496   23.440   28.366
H   -1.945   22.599   28.754
H   0.422   24.158   25.685
H   -1.393   24.537   25.581
H   -0.293   24.847   24.314
H   -0.183   18.398   21.709
H   -0.843   15.539   22.031
H   -2.619   18.689   22.252
H   -2.802   16.938   22.239
H   -2.124   17.747   20.718
H   1.484   16.624   21.201
H   1.345   15.070   21.703
H   2.097   16.595   24.103
H   3.057   17.419   22.905
H   3.216   15.665   23.091
H   -1.574   12.186   24.685
H   -1.680   11.080   26.093
H   -0.087   11.455   25.280
H   -2.216   13.549   30.657
H   0.905   11.174   32.893
H   -0.327   11.894   33.857
H   0.893   12.972   33.119
H   -0.084   16.256   37.490
H   -1.784   17.047   37.471
H   -0.122   19.111   37.785
H   -0.735   15.894   39.840
H   -0.710   16.900   41.325
H   0.735   16.445   40.748
H   -0.308   19.408   41.223
H   0.271   20.089   39.736
H   2.436   18.477   40.953
H   1.684   19.439   42.170
H   2.223   20.702   39.647
H   3.667   20.048   40.491
H   2.372   21.259   42.687
H   0.947   22.632   42.032
H   1.310   22.683   40.365
H   2.399   23.695   41.453
H   4.758   22.378   41.208
H   4.660   21.168   42.512
H   3.287   23.410   43.598
H   4.760   23.957   42.927
H   6.017   22.971   44.185
H   5.228   21.470   44.441
H   3.898   23.872   45.561
H   5.858   23.856   47.047
H   6.330   22.304   46.555
H   5.088   22.440   47.764
H   3.335   21.192   46.855
H   2.608   21.416   45.478
H   2.314   23.732   47.216
H   1.625   22.271   47.841
H   0.944   23.350   45.045
H   0.190   24.052   46.363
H   0.173   21.030   46.123
H   -0.024   21.671   43.719
H   -1.275   22.853   44.180
H   -1.676   21.182   44.309
H   -1.856   21.367   47.205
H   -2.418   22.658   46.100
H   -1.454   23.094   47.431

