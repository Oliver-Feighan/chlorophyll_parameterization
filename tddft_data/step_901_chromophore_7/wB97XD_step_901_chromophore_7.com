%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_901_chromophore_7 TDDFT with wB97XD functional

0 1
Mg   25.741   0.104   29.485
C   27.621   -0.284   32.462
C   22.988   0.266   31.404
C   23.940   0.120   26.631
C   28.718   -0.681   27.615
N   25.309   -0.168   31.734
C   26.276   -0.126   32.761
C   25.633   0.025   34.099
C   24.073   0.199   33.758
C   24.119   0.051   32.204
C   23.033   -0.688   34.559
C   26.264   1.262   34.854
C   25.632   1.806   36.140
C   26.627   1.912   37.336
O   27.731   2.464   37.355
O   26.231   1.121   38.314
N   23.738   0.207   29.104
C   22.709   0.195   29.992
C   21.375   0.264   29.384
C   21.630   0.241   27.959
C   23.174   0.292   27.842
C   20.096   0.200   30.171
C   20.650   0.254   26.809
O   20.912   0.289   25.604
C   19.214   0.404   27.162
N   26.241   -0.100   27.427
C   25.354   0.043   26.468
C   26.030   -0.152   25.081
C   27.544   -0.266   25.418
C   27.496   -0.409   26.930
C   25.503   -1.373   24.406
C   28.262   1.141   25.006
C   29.753   1.083   24.854
N   27.813   -0.319   29.885
C   28.876   -0.655   29.049
C   30.023   -0.957   29.821
C   29.636   -0.755   31.198
C   28.278   -0.391   31.160
C   31.334   -1.349   29.215
C   30.101   -0.622   32.563
O   31.174   -0.681   33.160
C   28.826   -0.244   33.423
C   28.829   -1.310   34.458
O   28.690   -2.500   34.293
O   28.968   -0.720   35.681
C   28.802   -1.671   36.806
C   27.021   0.952   39.498
C   26.153   0.987   40.711
C   26.465   1.174   42.022
C   27.955   1.459   42.491
C   25.428   1.197   43.166
C   24.053   0.481   43.039
C   22.829   1.409   43.334
C   22.313   1.249   44.731
C   21.378   0.001   44.842
C   21.692   2.534   45.370
C   22.691   3.380   46.061
C   22.436   3.417   47.541
C   23.057   4.707   48.153
C   23.473   4.489   49.598
C   21.963   5.863   48.143
C   22.472   7.151   47.536
C   21.547   7.822   46.484
C   22.045   7.980   45.026
C   21.514   9.313   44.443
C   21.598   6.842   44.020
H   22.103   0.506   31.997
H   23.373   0.145   25.698
H   29.543   -0.918   26.941
H   25.609   -0.874   34.715
H   23.935   1.244   34.034
H   23.411   -1.297   35.381
H   22.520   -1.333   33.846
H   22.328   0.005   35.018
H   26.393   1.989   34.053
H   27.286   0.947   35.065
H   24.769   1.212   36.441
H   25.191   2.745   35.808
H   19.370   1.009   30.089
H   20.332   0.155   31.234
H   19.567   -0.747   30.073
H   18.604   0.574   26.274
H   19.008   1.199   27.878
H   18.941   -0.592   27.509
H   25.756   0.721   24.489
H   28.111   -1.142   25.103
H   24.820   -1.914   25.060
H   26.306   -2.081   24.200
H   25.002   -1.155   23.462
H   27.960   2.031   25.559
H   27.770   1.373   24.062
H   30.099   0.105   24.520
H   30.236   1.249   25.816
H   30.190   1.845   24.208
H   31.426   -2.373   28.854
H   32.124   -1.157   29.942
H   31.441   -0.695   28.350
H   29.081   0.753   33.785
H   27.805   -1.419   37.168
H   29.463   -1.253   37.565
H   29.013   -2.719   36.591
H   27.692   1.779   39.725
H   27.595   0.026   39.453
H   25.132   0.785   40.386
H   28.177   0.704   43.244
H   28.160   2.337   43.105
H   28.666   1.485   41.665
H   25.290   2.247   43.422
H   25.867   0.708   44.036
H   24.068   -0.384   43.702
H   23.925   0.033   42.054
H   22.055   1.214   42.592
H   23.030   2.480   43.287
H   23.203   1.005   45.311
H   21.537   -0.717   44.038
H   20.326   0.267   44.937
H   21.683   -0.541   45.737
H   20.930   2.154   46.050
H   21.122   3.095   44.629
H   22.577   4.405   45.709
H   23.733   3.153   45.837
H   22.966   2.662   48.121
H   21.406   3.217   47.833
H   24.010   4.991   47.706
H   22.620   4.124   50.169
H   23.739   5.483   49.959
H   24.357   3.851   49.601
H   21.686   6.081   49.175
H   21.009   5.663   47.654
H   23.489   7.103   47.145
H   22.511   7.979   48.243
H   21.151   8.738   46.923
H   20.669   7.178   46.445
H   23.123   8.097   45.131
H   22.224   10.093   44.720
H   20.521   9.437   44.875
H   21.425   9.345   43.357
H   22.489   6.393   43.580
H   20.844   7.155   43.298
H   21.172   6.069   44.661

