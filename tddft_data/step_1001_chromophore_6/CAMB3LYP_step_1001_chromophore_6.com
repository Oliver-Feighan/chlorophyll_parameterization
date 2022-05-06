%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1001_chromophore_6 TDDFT with cam-b3lyp functional

0 1
Mg   17.157   -1.779   27.698
C   16.336   0.185   30.426
C   19.042   -3.795   29.847
C   18.359   -3.465   25.073
C   15.819   0.686   25.564
N   17.665   -1.770   29.894
C   17.079   -0.944   30.825
C   17.579   -1.342   32.204
C   18.560   -2.522   31.973
C   18.396   -2.728   30.460
C   20.074   -2.174   32.270
C   16.425   -1.651   33.192
C   16.421   -0.776   34.490
C   17.780   -0.480   35.187
O   18.388   0.528   35.069
O   18.227   -1.652   35.774
N   18.343   -3.468   27.527
C   18.973   -4.238   28.490
C   19.507   -5.475   27.860
C   19.378   -5.284   26.444
C   18.624   -4.056   26.347
C   19.853   -6.671   28.677
C   19.888   -6.164   25.331
O   19.605   -6.015   24.126
C   20.717   -7.329   25.682
N   16.970   -1.513   25.600
C   17.719   -2.256   24.825
C   17.667   -1.640   23.409
C   16.526   -0.517   23.514
C   16.377   -0.443   24.989
C   19.116   -1.041   22.874
C   15.161   -1.001   22.932
C   14.211   -0.059   22.250
N   16.168   0.031   27.897
C   15.660   0.917   26.977
C   14.938   1.988   27.681
C   15.244   1.768   29.028
C   16.005   0.546   29.126
C   14.229   3.103   27.113
C   15.052   2.230   30.346
O   14.411   3.179   30.759
C   15.873   1.290   31.338
C   16.901   2.106   32.055
O   17.995   2.392   31.684
O   16.423   2.621   33.255
C   17.245   3.497   34.097
C   19.705   -1.789   36.095
C   19.950   -1.332   37.511
C   21.054   -1.453   38.248
C   22.418   -1.944   37.729
C   21.097   -0.941   39.664
C   20.373   -1.911   40.625
C   20.938   -3.324   40.823
C   20.056   -4.416   40.164
C   19.058   -4.942   41.181
C   21.008   -5.586   39.609
C   22.205   -6.034   40.565
C   22.254   -7.598   40.691
C   23.692   -8.041   40.832
C   24.551   -7.766   39.567
C   23.862   -9.543   41.244
C   24.611   -9.536   42.605
C   26.127   -9.972   42.565
C   26.917   -9.311   43.727
C   27.551   -10.442   44.698
C   27.978   -8.344   43.196
H   19.568   -4.385   30.601
H   18.954   -3.738   24.199
H   15.400   1.424   24.877
H   18.068   -0.493   32.682
H   18.326   -3.442   32.509
H   20.350   -1.237   32.755
H   20.768   -2.209   31.430
H   20.393   -2.959   32.956
H   16.417   -2.727   33.365
H   15.505   -1.482   32.632
H   15.947   -1.272   35.337
H   15.761   0.087   34.401
H   19.393   -6.716   29.665
H   20.933   -6.532   28.647
H   19.723   -7.663   28.244
H   21.549   -7.078   26.341
H   21.050   -7.772   24.744
H   20.108   -8.118   26.125
H   17.358   -2.446   22.743
H   16.784   0.421   23.022
H   19.770   -1.049   23.746
H   19.082   -0.024   22.484
H   19.719   -1.617   22.172
H   14.474   -1.520   23.601
H   15.450   -1.660   22.114
H   13.232   -0.157   22.719
H   14.132   -0.224   21.176
H   14.554   0.956   22.449
H   13.496   2.656   26.443
H   14.820   3.838   26.566
H   13.712   3.623   27.919
H   15.175   0.762   31.988
H   17.749   2.938   34.885
H   16.419   4.123   34.433
H   17.911   4.059   33.441
H   20.389   -1.215   35.470
H   20.056   -2.821   36.069
H   19.134   -0.838   38.039
H   22.341   -2.418   36.751
H   22.745   -2.690   38.453
H   23.138   -1.125   37.732
H   20.567   0.007   39.569
H   22.134   -0.758   39.943
H   19.343   -1.980   40.274
H   20.387   -1.463   41.619
H   20.946   -3.562   41.887
H   21.927   -3.241   40.372
H   19.583   -4.000   39.274
H   19.056   -6.032   41.205
H   18.061   -4.726   40.798
H   19.195   -4.630   42.216
H   21.414   -5.030   38.764
H   20.501   -6.442   39.165
H   22.046   -5.613   41.558
H   23.110   -5.598   40.140
H   21.858   -7.930   39.731
H   21.851   -7.998   41.621
H   23.956   -7.328   41.612
H   25.215   -6.955   39.867
H   24.116   -7.420   38.629
H   25.333   -8.500   39.374
H   24.358   -10.082   40.437
H   22.875   -9.972   41.417
H   24.129   -10.327   43.180
H   24.436   -8.619   43.168
H   26.596   -9.805   41.596
H   26.204   -11.041   42.764
H   26.414   -8.698   44.475
H   28.094   -11.252   44.210
H   26.720   -10.935   45.202
H   28.169   -9.969   45.461
H   28.041   -8.254   42.112
H   28.934   -8.794   43.468
H   27.840   -7.395   43.714

