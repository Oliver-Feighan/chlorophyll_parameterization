%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_951_chromophore_26 TDDFT with cam-b3lyp functional

0 1
Mg   -8.780   18.352   43.171
C   -5.263   18.040   42.981
C   -8.306   21.664   42.141
C   -12.113   18.600   42.978
C   -9.023   14.867   43.291
N   -6.929   19.754   42.612
C   -5.592   19.409   42.747
C   -4.680   20.685   42.497
C   -5.756   21.752   42.359
C   -7.056   21.010   42.367
C   -5.660   22.890   43.460
C   -3.714   20.548   41.412
C   -4.028   19.612   40.154
C   -3.765   20.249   38.794
O   -2.616   20.116   38.295
O   -4.873   20.715   38.124
N   -10.082   19.956   42.638
C   -9.711   21.193   42.326
C   -10.891   22.070   42.239
C   -11.990   21.207   42.338
C   -11.396   19.845   42.703
C   -10.910   23.566   42.070
C   -13.470   21.504   42.060
O   -13.732   22.666   41.862
C   -14.669   20.452   41.948
N   -10.317   16.926   42.988
C   -11.614   17.272   43.042
C   -12.541   16.025   43.215
C   -11.534   14.849   43.139
C   -10.168   15.537   43.133
C   -13.559   15.982   44.287
C   -11.778   13.928   41.855
C   -13.167   13.500   41.477
N   -7.462   16.803   43.242
C   -7.661   15.415   43.358
C   -6.387   14.737   43.484
C   -5.438   15.724   43.404
C   -6.137   16.979   43.244
C   -6.208   13.243   43.642
C   -4.067   16.022   43.521
O   -3.174   15.213   43.598
C   -3.882   17.513   43.213
C   -3.032   18.118   44.287
O   -2.008   18.769   44.186
O   -3.702   17.776   45.435
C   -2.874   17.974   46.609
C   -4.619   20.921   36.688
C   -5.844   20.835   35.747
C   -6.121   19.886   34.766
C   -5.225   18.612   34.580
C   -7.135   20.142   33.678
C   -8.470   19.483   33.989
C   -9.413   19.650   32.741
C   -10.825   20.239   33.063
C   -11.924   19.122   33.167
C   -11.266   21.357   32.077
C   -10.704   22.782   32.377
C   -9.662   23.246   31.357
C   -10.148   24.368   30.456
C   -9.491   25.707   30.747
C   -10.106   23.934   28.976
C   -11.205   22.896   28.518
C   -11.891   23.329   27.136
C   -13.190   24.179   27.314
C   -14.499   23.320   27.095
C   -13.139   25.413   26.363
H   -8.207   22.692   41.784
H   -13.197   18.540   42.863
H   -9.113   13.784   43.402
H   -4.073   20.844   43.389
H   -5.759   22.339   41.441
H   -4.668   22.984   43.902
H   -6.443   22.739   44.203
H   -5.863   23.858   43.002
H   -2.779   20.104   41.754
H   -3.513   21.592   41.168
H   -5.097   19.423   40.255
H   -3.518   18.663   40.322
H   -10.930   23.787   41.003
H   -10.084   23.969   42.657
H   -11.840   23.945   42.492
H   -14.946   20.242   42.982
H   -14.456   19.581   41.328
H   -15.530   21.040   41.630
H   -13.015   15.999   42.234
H   -11.584   14.160   43.983
H   -14.563   16.014   43.865
H   -13.480   16.853   44.937
H   -13.411   15.025   44.788
H   -11.277   12.971   42.006
H   -11.272   14.442   41.038
H   -13.285   12.465   41.157
H   -13.521   14.072   40.619
H   -13.861   13.601   42.312
H   -7.113   12.858   43.171
H   -6.199   12.957   44.694
H   -5.350   12.873   43.082
H   -3.234   17.633   42.344
H   -3.438   17.631   47.476
H   -2.685   18.989   46.958
H   -1.965   17.375   46.674
H   -3.787   20.316   36.326
H   -4.282   21.957   36.702
H   -6.382   21.745   35.483
H   -4.945   18.059   35.477
H   -4.263   19.009   34.255
H   -5.654   17.947   33.830
H   -6.716   19.778   32.740
H   -7.324   21.214   33.610
H   -8.854   19.886   34.926
H   -8.494   18.429   34.266
H   -9.487   18.691   32.228
H   -8.852   20.261   32.034
H   -10.870   20.584   34.097
H   -12.165   18.815   32.149
H   -12.852   19.557   33.540
H   -11.583   18.354   33.862
H   -12.352   21.336   32.160
H   -10.998   20.963   31.097
H   -10.215   22.760   33.350
H   -11.551   23.468   32.413
H   -9.295   22.405   30.769
H   -8.766   23.499   31.923
H   -11.180   24.700   30.576
H   -8.665   25.613   31.451
H   -10.144   26.436   31.226
H   -9.176   26.173   29.813
H   -9.117   23.640   28.626
H   -10.398   24.810   28.397
H   -12.038   22.762   29.208
H   -10.780   21.899   28.405
H   -11.989   22.364   26.638
H   -11.141   23.939   26.633
H   -13.268   24.578   28.326
H   -14.273   22.275   27.309
H   -14.763   23.516   26.056
H   -15.306   23.523   27.798
H   -12.991   24.993   25.367
H   -12.308   26.028   26.709
H   -14.111   25.905   26.371

