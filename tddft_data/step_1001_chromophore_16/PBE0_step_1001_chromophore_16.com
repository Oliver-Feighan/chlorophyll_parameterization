%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1001_chromophore_16 TDDFT with PBE1PBE functional

0 1
Mg   40.624   41.756   26.513
C   39.946   43.854   29.392
C   41.700   39.325   28.806
C   41.490   39.676   23.908
C   40.002   44.345   24.460
N   40.854   41.704   28.839
C   40.389   42.557   29.813
C   40.387   42.056   31.243
C   40.984   40.601   31.056
C   41.185   40.529   29.467
C   42.321   40.449   31.792
C   38.964   42.029   31.889
C   38.813   41.194   33.125
C   38.105   41.884   34.322
O   37.120   42.558   34.259
O   38.822   41.830   35.535
N   41.551   39.856   26.408
C   41.845   39.037   27.466
C   42.213   37.714   26.993
C   42.179   37.728   25.546
C   41.811   39.137   25.209
C   42.553   36.513   27.941
C   42.523   36.696   24.537
O   42.696   36.935   23.308
C   42.734   35.234   25.065
N   40.691   41.989   24.487
C   40.987   40.968   23.564
C   41.118   41.534   22.142
C   40.350   42.913   22.339
C   40.437   43.156   23.879
C   42.615   41.647   21.783
C   38.905   42.770   21.936
C   38.276   44.078   21.338
N   39.965   43.633   26.778
C   39.678   44.607   25.841
C   39.200   45.838   26.473
C   39.268   45.595   27.859
C   39.678   44.258   28.019
C   38.968   47.135   25.720
C   39.062   46.190   29.189
O   38.813   47.342   29.541
C   39.343   45.011   30.160
C   40.337   45.477   31.129
O   41.478   45.720   30.918
O   39.689   45.850   32.301
C   40.528   46.434   33.362
C   38.265   42.471   36.671
C   38.873   41.913   37.886
C   38.793   42.450   39.061
C   38.071   43.710   39.485
C   39.539   41.863   40.213
C   38.636   41.155   41.287
C   39.068   39.704   41.590
C   38.551   39.329   43.009
C   39.617   38.535   43.823
C   37.256   38.495   42.986
C   36.557   38.264   44.308
C   36.520   36.754   44.666
C   35.493   36.409   45.801
C   35.980   35.277   46.677
C   34.095   35.946   45.285
C   32.946   36.490   46.132
C   31.875   35.312   46.347
C   31.754   34.804   47.771
C   30.749   35.638   48.629
C   31.504   33.256   47.793
H   41.647   38.570   29.593
H   41.622   38.944   23.108
H   39.722   45.128   23.751
H   41.063   42.629   31.878
H   40.490   39.720   31.465
H   43.091   40.271   31.040
H   42.159   39.600   32.456
H   42.589   41.160   32.573
H   38.288   41.609   31.144
H   38.708   43.045   32.189
H   39.770   40.785   33.448
H   38.210   40.322   32.870
H   43.531   36.170   27.601
H   41.859   35.673   27.950
H   42.616   36.788   28.994
H   41.877   34.893   25.646
H   43.685   35.168   25.592
H   42.841   34.619   24.171
H   40.575   40.869   21.470
H   40.813   43.759   21.830
H   43.175   41.088   22.532
H   42.948   42.684   21.795
H   42.839   41.119   20.856
H   38.403   42.421   22.838
H   38.670   42.051   21.152
H   37.600   43.697   20.572
H   39.002   44.827   21.024
H   37.605   44.528   22.069
H   38.991   48.045   26.319
H   37.985   47.165   25.250
H   39.792   47.247   25.014
H   38.503   44.641   30.748
H   41.428   45.832   33.483
H   39.968   46.486   34.296
H   40.741   47.489   33.189
H   37.207   42.213   36.717
H   38.382   43.554   36.638
H   39.457   41.002   37.752
H   37.703   44.344   38.678
H   38.839   44.318   39.962
H   37.333   43.588   40.278
H   39.968   42.752   40.678
H   40.365   41.186   39.994
H   37.656   41.269   40.823
H   38.623   41.811   42.158
H   40.123   39.497   41.415
H   38.505   39.128   40.855
H   38.275   40.230   43.558
H   40.306   37.951   43.211
H   39.119   37.833   44.492
H   40.206   39.188   44.466
H   37.438   37.562   42.453
H   36.476   39.064   42.481
H   35.510   38.529   44.162
H   36.997   38.803   45.146
H   37.413   36.197   44.949
H   36.157   36.304   43.742
H   35.454   37.300   46.428
H   35.123   34.763   47.113
H   36.525   35.663   47.539
H   36.654   34.573   46.190
H   34.275   34.873   45.219
H   33.992   36.334   44.272
H   32.350   37.225   45.591
H   33.313   36.801   47.110
H   32.025   34.484   45.654
H   30.901   35.763   46.154
H   32.768   34.969   48.135
H   30.331   36.459   48.047
H   31.272   36.162   49.429
H   29.948   35.014   49.027
H   32.338   32.835   47.232
H   30.568   32.928   47.339
H   31.293   32.908   48.804

