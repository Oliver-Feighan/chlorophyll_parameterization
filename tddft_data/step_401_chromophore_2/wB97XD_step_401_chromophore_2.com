%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_401_chromophore_2 TDDFT with wB97XD functional

0 1
Mg   3.078   -0.600   44.505
C   6.391   0.376   43.566
C   1.852   2.045   42.735
C   0.082   -2.139   44.522
C   4.580   -3.609   45.943
N   4.010   1.129   43.343
C   5.358   1.311   43.100
C   5.560   2.590   42.191
C   4.119   3.251   42.175
C   3.227   2.096   42.831
C   3.919   4.566   42.917
C   6.011   2.163   40.822
C   7.409   2.593   40.290
C   7.409   3.292   38.878
O   7.959   4.348   38.566
O   6.966   2.370   37.935
N   1.174   -0.044   43.912
C   0.908   1.100   43.171
C   -0.520   1.208   43.030
C   -1.122   -0.012   43.521
C   0.067   -0.839   44.024
C   -1.242   2.363   42.199
C   -2.567   -0.369   43.594
O   -3.468   0.283   43.082
C   -3.052   -1.733   44.231
N   2.512   -2.560   45.144
C   1.185   -2.915   45.021
C   0.924   -4.365   45.536
C   2.362   -4.870   45.781
C   3.240   -3.597   45.633
C   0.172   -4.320   46.921
C   2.779   -6.006   44.717
C   2.677   -7.432   45.324
N   5.110   -1.491   44.774
C   5.505   -2.623   45.465
C   6.917   -2.547   45.612
C   7.335   -1.450   44.831
C   6.163   -0.802   44.395
C   7.732   -3.466   46.431
C   8.482   -0.723   44.320
O   9.645   -0.873   44.534
C   7.964   0.414   43.444
C   8.544   1.631   43.974
O   9.316   2.358   43.352
O   8.026   1.972   45.222
C   8.243   3.278   45.763
C   7.080   2.763   36.536
C   5.864   2.310   35.899
C   5.430   1.518   34.878
C   6.446   0.952   33.839
C   3.982   1.193   34.901
C   3.033   2.205   34.321
C   2.095   1.570   33.291
C   0.642   1.814   33.674
C   -0.289   0.563   33.455
C   0.043   3.033   32.992
C   -0.118   2.957   31.476
C   -1.579   2.829   31.079
C   -2.065   4.027   30.204
C   -2.814   5.011   31.099
C   -3.004   3.494   28.983
C   -2.276   3.472   27.572
C   -2.694   2.307   26.716
C   -3.175   2.664   25.341
C   -4.554   3.394   25.353
C   -3.196   1.465   24.393
H   1.418   2.845   42.132
H   -0.817   -2.729   44.711
H   4.899   -4.519   46.454
H   6.367   3.132   42.684
H   3.786   3.432   41.153
H   4.770   4.940   43.486
H   3.004   4.421   43.492
H   3.764   5.302   42.128
H   5.294   2.451   40.053
H   5.956   1.074   40.803
H   8.115   1.764   40.235
H   7.678   3.379   40.997
H   -1.891   1.839   41.497
H   -0.564   3.091   41.753
H   -1.882   3.029   42.778
H   -4.116   -1.668   44.460
H   -2.625   -1.962   45.208
H   -2.864   -2.475   43.455
H   0.339   -4.977   44.850
H   2.476   -5.268   46.789
H   -0.716   -3.790   46.575
H   0.613   -3.630   47.640
H   -0.106   -5.247   47.422
H   3.806   -5.746   44.458
H   2.138   -5.937   43.839
H   2.202   -7.378   46.303
H   3.680   -7.836   45.465
H   2.073   -7.979   44.600
H   7.165   -3.555   47.357
H   8.757   -3.152   46.628
H   7.855   -4.431   45.940
H   8.261   0.407   42.395
H   7.417   3.955   45.543
H   9.146   3.708   45.330
H   8.247   3.243   46.853
H   7.882   2.245   36.010
H   7.093   3.799   36.200
H   5.111   2.390   36.683
H   6.380   1.391   32.843
H   6.131   -0.089   33.772
H   7.505   1.029   34.084
H   3.691   1.048   35.942
H   3.968   0.232   34.388
H   3.671   2.934   33.820
H   2.506   2.807   35.061
H   2.314   0.533   33.033
H   2.363   2.082   32.367
H   0.590   2.054   34.735
H   0.175   -0.275   32.933
H   -1.211   0.973   33.043
H   -0.508   0.260   34.479
H   0.576   3.961   33.200
H   -0.961   3.111   33.409
H   0.463   2.068   31.231
H   0.358   3.861   31.098
H   -2.184   2.521   31.932
H   -1.598   1.911   30.492
H   -1.259   4.639   29.799
H   -2.767   5.988   30.618
H   -2.438   5.113   32.117
H   -3.863   4.715   31.133
H   -3.725   4.304   28.873
H   -3.567   2.627   29.326
H   -1.205   3.321   27.708
H   -2.471   4.442   27.116
H   -3.315   1.519   27.142
H   -1.721   1.856   26.520
H   -2.497   3.373   24.866
H   -5.306   2.637   25.133
H   -4.657   4.251   24.688
H   -4.703   3.616   26.410
H   -3.902   1.524   23.564
H   -3.307   0.615   25.066
H   -2.234   1.183   23.964

