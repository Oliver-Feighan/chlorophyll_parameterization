%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_501_chromophore_26 TDDFT with PBE1PBE functional

0 1
Mg   -9.665   18.334   42.784
C   -6.240   17.805   42.496
C   -9.099   21.676   42.053
C   -12.912   18.874   42.483
C   -10.148   14.914   43.005
N   -7.916   19.561   42.171
C   -6.598   19.162   42.258
C   -5.633   20.296   42.076
C   -6.573   21.558   42.319
C   -7.951   20.936   42.136
C   -6.371   22.541   43.568
C   -4.814   20.285   40.703
C   -5.518   19.697   39.444
C   -5.279   20.481   38.176
O   -4.441   21.398   38.034
O   -5.957   19.873   37.138
N   -10.857   20.015   42.214
C   -10.420   21.322   41.965
C   -11.510   22.180   41.589
C   -12.666   21.369   41.684
C   -12.206   20.014   42.111
C   -11.295   23.689   41.462
C   -14.079   21.930   41.405
O   -14.258   23.081   40.994
C   -15.201   21.030   41.712
N   -11.248   17.013   42.634
C   -12.523   17.529   42.794
C   -13.590   16.437   43.028
C   -12.642   15.171   42.946
C   -11.285   15.702   42.806
C   -14.416   16.471   44.275
C   -12.973   14.219   41.688
C   -13.020   12.771   42.101
N   -8.457   16.645   42.958
C   -8.803   15.309   43.041
C   -7.581   14.474   43.089
C   -6.530   15.433   42.878
C   -7.129   16.696   42.805
C   -7.420   13.009   43.037
C   -5.087   15.659   42.649
O   -4.194   14.852   42.532
C   -4.844   17.188   42.437
C   -3.866   17.686   43.405
O   -2.757   18.189   43.223
O   -4.284   17.228   44.682
C   -3.453   17.676   45.820
C   -5.395   20.301   35.861
C   -6.338   20.015   34.748
C   -6.658   18.747   34.412
C   -6.187   17.454   35.073
C   -7.666   18.500   33.332
C   -9.170   18.770   33.825
C   -10.167   17.850   33.062
C   -11.318   18.705   32.401
C   -12.701   18.181   32.750
C   -11.159   18.969   30.885
C   -11.258   20.482   30.507
C   -9.960   21.240   30.542
C   -10.013   22.592   31.223
C   -9.159   22.602   32.554
C   -9.416   23.742   30.227
C   -10.288   25.073   30.390
C   -11.306   25.400   29.230
C   -12.494   24.397   29.212
C   -12.214   23.359   28.137
C   -13.902   25.045   29.078
H   -8.861   22.742   42.075
H   -13.994   18.961   42.601
H   -10.406   13.864   43.154
H   -4.852   20.252   42.835
H   -6.356   22.137   41.421
H   -5.495   22.348   44.187
H   -7.182   22.500   44.295
H   -6.141   23.542   43.202
H   -3.840   19.817   40.842
H   -4.553   21.324   40.502
H   -6.590   19.594   39.615
H   -5.141   18.694   39.246
H   -11.949   24.167   40.732
H   -10.297   23.875   41.066
H   -11.499   24.100   42.451
H   -15.272   20.923   42.794
H   -15.056   20.075   41.206
H   -16.084   21.543   41.331
H   -14.188   16.368   42.119
H   -12.693   14.777   43.961
H   -14.929   15.510   44.316
H   -15.203   17.208   44.115
H   -13.792   16.695   45.141
H   -12.210   14.427   40.938
H   -13.867   14.454   41.109
H   -12.158   12.271   41.660
H   -13.818   12.222   41.600
H   -12.898   12.691   43.181
H   -8.344   12.430   43.001
H   -6.931   12.731   43.970
H   -6.969   12.899   42.051
H   -4.468   17.264   41.417
H   -4.210   17.982   46.543
H   -2.740   18.493   45.709
H   -3.066   16.752   46.250
H   -4.493   19.733   35.636
H   -5.175   21.368   35.818
H   -6.644   20.844   34.109
H   -5.615   16.844   34.375
H   -7.119   16.941   35.309
H   -5.622   17.630   35.988
H   -7.613   17.560   32.783
H   -7.573   19.269   32.565
H   -9.323   19.797   33.495
H   -9.185   18.698   34.912
H   -10.543   17.140   33.798
H   -9.757   17.139   32.345
H   -11.355   19.711   32.819
H   -12.755   17.093   32.770
H   -13.436   18.650   32.096
H   -12.910   18.456   33.784
H   -12.001   18.519   30.360
H   -10.251   18.489   30.520
H   -11.892   21.024   31.209
H   -11.620   20.539   29.481
H   -9.611   21.285   29.511
H   -9.193   20.630   31.019
H   -11.024   22.868   31.524
H   -8.644   23.530   32.801
H   -8.419   21.805   32.471
H   -9.765   22.395   33.436
H   -9.205   23.539   29.177
H   -8.457   24.076   30.624
H   -9.645   25.948   30.492
H   -10.932   25.026   31.268
H   -10.628   25.330   28.380
H   -11.654   26.429   29.325
H   -12.638   23.929   30.186
H   -11.528   23.706   27.365
H   -13.161   23.155   27.636
H   -11.807   22.409   28.485
H   -14.571   24.755   29.888
H   -14.549   24.716   28.265
H   -13.903   26.132   29.000

