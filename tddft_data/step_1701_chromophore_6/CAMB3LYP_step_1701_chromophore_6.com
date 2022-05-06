%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1701_chromophore_6 TDDFT with cam-b3lyp functional

0 1
Mg   17.927   -1.786   27.707
C   16.920   0.287   30.333
C   19.982   -3.555   29.724
C   18.940   -3.557   24.892
C   16.261   0.448   25.606
N   18.495   -1.518   29.744
C   17.829   -0.807   30.707
C   18.230   -1.173   32.143
C   19.406   -2.229   31.888
C   19.379   -2.423   30.354
C   20.759   -1.807   32.325
C   17.027   -1.625   32.970
C   16.925   -1.252   34.468
C   18.110   -0.491   35.020
O   18.230   0.731   35.062
O   19.163   -1.373   35.360
N   19.214   -3.426   27.376
C   19.850   -4.092   28.430
C   20.443   -5.290   27.862
C   20.235   -5.303   26.430
C   19.412   -4.047   26.180
C   21.207   -6.233   28.687
C   20.726   -6.227   25.283
O   20.428   -6.034   24.093
C   21.593   -7.351   25.580
N   17.745   -1.481   25.529
C   18.182   -2.390   24.618
C   17.916   -1.946   23.123
C   16.843   -0.811   23.437
C   16.982   -0.549   24.930
C   19.184   -1.450   22.414
C   15.457   -1.301   23.155
C   14.503   -0.283   22.472
N   16.846   -0.085   27.925
C   16.133   0.656   27.017
C   15.405   1.776   27.631
C   15.730   1.657   28.979
C   16.558   0.513   29.066
C   14.589   2.823   27.030
C   15.608   2.261   30.303
O   15.015   3.297   30.674
C   16.327   1.322   31.287
C   17.318   2.032   32.115
O   18.545   2.189   31.899
O   16.645   2.670   33.134
C   17.294   3.536   34.194
C   20.352   -0.646   35.807
C   21.337   -1.679   36.388
C   21.380   -2.188   37.630
C   20.403   -1.628   38.724
C   22.567   -3.090   38.041
C   22.204   -4.517   38.332
C   22.633   -4.962   39.743
C   22.863   -6.507   39.967
C   21.529   -7.239   40.395
C   23.806   -6.695   41.230
C   25.125   -7.420   40.802
C   24.957   -8.990   40.978
C   26.206   -9.511   41.619
C   26.560   -10.919   41.072
C   26.139   -9.515   43.182
C   27.351   -8.935   43.832
C   27.505   -9.507   45.209
C   28.516   -10.721   45.192
C   29.891   -10.293   45.722
C   27.851   -11.810   46.108
H   20.501   -4.142   30.485
H   19.286   -4.082   23.999
H   15.627   1.113   25.015
H   18.719   -0.358   32.678
H   19.131   -3.211   32.272
H   20.762   -0.816   32.779
H   21.418   -1.718   31.462
H   21.192   -2.467   33.077
H   16.958   -2.710   32.891
H   16.081   -1.362   32.496
H   16.742   -2.166   35.032
H   15.986   -0.720   34.624
H   22.256   -6.305   28.400
H   20.726   -7.203   28.561
H   21.208   -5.982   29.748
H   22.045   -7.777   24.685
H   20.874   -8.058   25.994
H   22.272   -7.275   26.429
H   17.476   -2.751   22.533
H   17.106   0.030   22.796
H   19.147   -2.011   21.481
H   20.059   -1.720   23.005
H   19.065   -0.406   22.125
H   14.920   -1.621   24.048
H   15.384   -2.143   22.467
H   14.543   -0.276   21.383
H   14.478   0.753   22.811
H   13.496   -0.668   22.626
H   14.383   2.504   26.009
H   15.196   3.727   26.995
H   13.753   3.148   27.650
H   15.575   0.791   31.870
H   17.133   4.584   33.940
H   18.377   3.411   34.174
H   16.983   3.321   35.217
H   20.040   0.141   36.493
H   20.757   -0.200   34.898
H   22.057   -2.073   35.671
H   19.791   -2.416   39.162
H   19.683   -0.895   38.360
H   20.968   -1.177   39.539
H   22.979   -2.535   38.884
H   23.203   -3.174   37.160
H   22.437   -5.155   37.479
H   21.115   -4.489   38.320
H   22.055   -4.428   40.497
H   23.616   -4.519   39.902
H   23.324   -6.993   39.107
H   20.926   -6.433   40.814
H   21.652   -8.042   41.121
H   21.125   -7.673   39.481
H   23.321   -7.179   42.077
H   24.193   -5.777   41.672
H   25.851   -7.042   41.523
H   25.424   -7.251   39.767
H   24.814   -9.361   39.963
H   24.024   -9.256   41.474
H   27.023   -8.910   41.219
H   25.656   -11.403   40.705
H   27.032   -11.510   41.857
H   27.323   -10.880   40.295
H   26.031   -10.523   43.582
H   25.299   -8.865   43.428
H   27.129   -7.870   43.892
H   28.255   -9.067   43.237
H   26.532   -9.850   45.560
H   27.809   -8.720   45.898
H   28.665   -11.114   44.186
H   29.768   -9.210   45.720
H   30.737   -10.811   45.271
H   29.954   -10.527   46.785
H   27.478   -12.580   45.432
H   27.165   -11.396   46.846
H   28.535   -12.428   46.690

