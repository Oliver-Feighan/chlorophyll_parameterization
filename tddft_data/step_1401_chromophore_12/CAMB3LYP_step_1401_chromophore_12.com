%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1401_chromophore_12 TDDFT with cam-b3lyp functional

0 1
Mg   48.061   15.063   27.781
C   46.083   15.184   30.691
C   50.273   17.315   29.394
C   49.955   15.167   25.053
C   45.620   13.140   26.208
N   48.202   16.141   29.794
C   47.251   15.944   30.850
C   47.848   16.653   32.126
C   48.976   17.579   31.598
C   49.176   17.007   30.199
C   48.613   19.115   31.502
C   48.315   15.542   33.226
C   48.013   15.887   34.668
C   49.039   15.400   35.701
O   49.896   14.560   35.494
O   48.849   16.091   36.907
N   49.862   16.173   27.221
C   50.619   16.998   28.081
C   51.868   17.448   27.363
C   51.774   16.727   26.076
C   50.457   16.071   26.016
C   52.865   18.297   28.019
C   52.809   16.697   25.039
O   52.640   16.120   23.952
C   54.145   17.351   25.314
N   47.770   14.481   25.921
C   48.740   14.516   24.954
C   48.505   13.453   23.817
C   47.138   12.791   24.198
C   46.838   13.463   25.548
C   48.531   14.096   22.347
C   47.193   11.277   24.430
C   48.078   10.476   23.354
N   46.151   14.396   28.224
C   45.267   13.603   27.526
C   44.080   13.392   28.280
C   44.358   13.989   29.490
C   45.663   14.549   29.484
C   42.826   12.700   27.793
C   43.771   14.151   30.814
O   42.740   13.771   31.302
C   44.920   14.938   31.695
C   44.339   16.181   32.293
O   43.842   17.122   31.672
O   44.408   16.048   33.662
C   44.078   17.230   34.460
C   50.080   16.239   37.742
C   50.056   17.629   38.307
C   51.097   18.358   38.798
C   52.557   17.845   38.982
C   50.840   19.739   39.384
C   51.550   20.932   38.678
C   52.830   21.414   39.414
C   53.871   22.178   38.514
C   55.180   22.335   39.453
C   53.384   23.605   38.063
C   53.884   23.901   36.615
C   52.675   24.366   35.644
C   53.046   24.107   34.154
C   54.013   25.228   33.706
C   51.792   24.096   33.274
C   52.156   23.662   31.835
C   50.964   23.028   31.101
C   50.726   23.656   29.664
C   49.727   24.802   29.847
C   50.091   22.599   28.695
H   50.922   18.081   29.824
H   50.618   15.050   24.193
H   45.134   12.376   25.597
H   47.111   17.292   32.613
H   49.868   17.476   32.216
H   48.646   19.462   30.470
H   49.371   19.783   31.911
H   47.742   19.400   32.093
H   49.377   15.405   33.022
H   47.878   14.588   32.935
H   47.090   15.430   35.024
H   47.852   16.949   34.848
H   53.625   17.787   28.611
H   52.361   18.946   28.735
H   53.448   18.961   27.381
H   54.899   16.992   24.614
H   54.457   17.059   26.317
H   54.088   18.419   25.101
H   49.298   12.713   23.924
H   46.466   13.057   23.383
H   48.725   13.295   21.634
H   49.337   14.821   22.232
H   47.590   14.620   22.180
H   46.190   10.856   24.359
H   47.468   10.888   25.410
H   48.578   11.089   22.605
H   47.372   9.845   22.814
H   48.780   9.807   23.851
H   42.133   12.434   28.592
H   43.114   11.857   27.166
H   42.350   13.415   27.122
H   45.082   14.215   32.495
H   44.342   18.187   34.010
H   44.635   17.129   35.392
H   42.997   17.126   34.557
H   50.983   16.144   37.139
H   50.241   15.467   38.495
H   49.033   18.004   38.347
H   52.717   16.766   38.978
H   52.914   18.043   39.992
H   53.248   18.291   38.267
H   51.132   19.738   40.434
H   49.774   19.966   39.386
H   50.886   21.796   38.648
H   51.825   20.772   37.635
H   53.292   20.607   39.981
H   52.513   22.185   40.116
H   53.950   21.510   37.657
H   54.847   22.231   40.486
H   55.627   23.310   39.261
H   56.017   21.653   39.305
H   53.835   24.374   38.690
H   52.310   23.787   38.114
H   54.454   23.084   36.172
H   54.581   24.732   36.724
H   52.727   25.442   35.805
H   51.712   23.991   35.991
H   53.595   23.169   34.240
H   54.478   25.783   34.521
H   53.652   25.917   32.943
H   54.781   24.612   33.238
H   51.379   25.100   33.371
H   51.066   23.430   33.739
H   52.968   22.937   31.888
H   52.578   24.458   31.222
H   50.048   23.190   31.670
H   51.107   21.955   30.973
H   51.623   23.974   29.133
H   49.842   25.467   28.991
H   49.770   25.330   30.799
H   48.816   24.204   29.826
H   49.388   23.040   27.989
H   49.777   21.694   29.216
H   50.869   22.128   28.095

