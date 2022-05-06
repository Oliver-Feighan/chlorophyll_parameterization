%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_201_chromophore_21 TDDFT with blyp functional

0 1
Mg   15.751   52.569   25.114
C   17.431   51.395   27.930
C   13.210   53.508   27.109
C   14.198   53.417   22.325
C   18.602   51.405   23.137
N   15.473   52.562   27.180
C   16.313   52.110   28.150
C   15.742   52.468   29.551
C   14.307   52.815   29.291
C   14.301   52.943   27.731
C   13.280   51.755   29.705
C   16.665   53.527   30.381
C   17.069   53.079   31.775
C   16.071   52.467   32.827
O   16.101   51.317   33.284
O   15.068   53.364   32.995
N   13.902   53.364   24.710
C   13.035   53.768   25.684
C   11.863   54.386   25.104
C   12.183   54.392   23.691
C   13.475   53.727   23.499
C   10.622   54.791   25.866
C   11.436   54.998   22.645
O   11.880   55.012   21.492
C   10.079   55.678   22.928
N   16.443   52.599   23.050
C   15.514   52.940   22.088
C   16.058   52.668   20.704
C   17.517   52.194   20.970
C   17.532   52.035   22.483
C   15.292   51.509   19.882
C   18.685   53.133   20.540
C   19.886   52.531   19.775
N   17.668   51.594   25.383
C   18.617   51.142   24.557
C   19.604   50.354   25.299
C   19.206   50.487   26.609
C   17.994   51.147   26.617
C   20.737   49.615   24.644
C   19.520   50.122   28.004
O   20.506   49.553   28.437
C   18.367   50.691   28.922
C   17.710   49.718   29.785
O   16.736   49.020   29.511
O   18.333   49.702   31.030
C   17.698   48.866   32.060
C   14.145   52.822   34.000
C   13.120   53.840   34.363
C   12.821   54.428   35.513
C   13.521   54.235   36.809
C   11.745   55.506   35.498
C   10.380   55.265   36.233
C   9.164   55.614   35.439
C   8.043   56.266   36.354
C   6.672   55.552   36.104
C   7.775   57.830   36.124
C   8.777   58.706   36.908
C   8.111   60.135   36.944
C   8.901   61.038   36.036
C   9.994   61.751   36.917
C   7.992   61.968   35.172
C   7.965   61.714   33.634
C   8.833   62.631   32.760
C   10.317   62.049   32.626
C   11.241   62.865   33.596
C   10.726   62.099   31.104
H   12.404   53.931   27.712
H   13.657   53.649   21.405
H   19.434   51.010   22.551
H   15.872   51.642   30.250
H   14.026   53.819   29.612
H   12.691   52.137   30.538
H   13.732   50.839   30.084
H   12.668   51.550   28.827
H   16.139   54.479   30.454
H   17.586   53.751   29.843
H   17.323   54.011   32.281
H   17.987   52.513   31.620
H   10.369   55.849   25.794
H   10.653   54.669   26.949
H   9.689   54.308   25.574
H   10.419   56.391   23.678
H   9.216   55.037   23.109
H   9.897   56.212   21.996
H   16.014   53.665   20.266
H   17.687   51.256   20.440
H   15.641   50.525   20.194
H   15.283   51.686   18.806
H   14.281   51.594   20.281
H   19.099   53.724   21.357
H   18.198   53.811   19.840
H   19.775   51.469   19.558
H   20.788   52.719   20.358
H   20.013   53.029   18.813
H   21.635   50.159   24.935
H   20.679   49.470   23.566
H   20.822   48.652   25.148
H   18.858   51.449   29.533
H   18.384   48.026   32.168
H   16.710   48.457   31.851
H   17.743   49.453   32.977
H   14.757   52.621   34.879
H   13.666   51.932   33.591
H   12.683   54.213   33.437
H   13.158   55.039   37.450
H   14.590   54.241   36.596
H   13.079   53.331   37.229
H   11.622   55.929   34.501
H   12.186   56.394   35.951
H   10.530   55.869   37.128
H   10.308   54.263   36.654
H   8.905   54.760   34.813
H   9.222   56.349   34.636
H   8.256   56.138   37.415
H   5.860   56.147   35.686
H   6.309   55.372   37.116
H   6.663   54.642   35.503
H   6.765   58.023   36.485
H   7.606   58.133   35.091
H   9.796   58.723   36.522
H   8.709   58.369   37.942
H   8.080   60.400   38.001
H   7.029   60.148   36.810
H   9.560   60.538   35.327
H   10.282   61.333   37.882
H   9.726   62.802   37.026
H   10.875   61.822   36.279
H   8.441   62.960   35.205
H   6.965   62.021   35.532
H   6.969   61.546   33.224
H   8.400   60.728   33.468
H   8.896   63.592   33.270
H   8.405   62.714   31.761
H   10.382   60.982   32.837
H   10.623   63.257   34.404
H   11.713   63.721   33.114
H   11.945   62.157   34.032
H   10.094   62.681   30.433
H   11.030   61.105   30.775
H   11.700   62.586   31.059

