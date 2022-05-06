%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_951_chromophore_1 TDDFT with PBE1PBE functional

0 1
Mg   -1.258   17.643   27.029
C   -1.800   15.408   29.610
C   -2.171   20.202   28.972
C   -1.201   19.762   24.265
C   -0.777   14.901   24.794
N   -1.890   17.755   29.000
C   -1.903   16.761   29.945
C   -2.234   17.352   31.352
C   -2.954   18.722   30.917
C   -2.287   18.930   29.566
C   -4.515   18.750   30.871
C   -0.968   17.544   32.231
C   -1.204   17.328   33.742
C   -0.034   16.745   34.624
O   0.856   16.073   34.194
O   -0.194   17.211   35.882
N   -1.419   19.774   26.772
C   -1.811   20.600   27.722
C   -1.593   21.977   27.249
C   -1.238   21.906   25.853
C   -1.234   20.456   25.555
C   -1.756   23.268   28.104
C   -0.956   22.945   24.938
O   -0.662   22.706   23.757
C   -0.993   24.379   25.364
N   -1.110   17.423   24.756
C   -1.167   18.417   23.871
C   -1.223   17.856   22.468
C   -0.798   16.344   22.651
C   -0.953   16.141   24.143
C   -2.449   18.152   21.598
C   0.669   16.061   22.095
C   1.898   16.295   23.023
N   -1.231   15.636   27.123
C   -1.069   14.607   26.158
C   -1.280   13.274   26.746
C   -1.470   13.559   28.112
C   -1.504   14.953   28.328
C   -1.250   12.032   25.985
C   -1.699   12.972   29.331
O   -1.697   11.766   29.687
C   -1.951   14.110   30.347
C   -1.066   13.900   31.534
O   0.133   14.045   31.543
O   -1.760   13.350   32.612
C   -0.847   13.001   33.801
C   0.845   16.771   36.858
C   0.488   17.433   38.175
C   0.352   16.909   39.414
C   0.653   15.546   39.966
C   -0.341   17.803   40.448
C   0.743   18.721   41.036
C   0.229   20.107   41.079
C   1.179   21.318   40.727
C   1.242   21.535   39.188
C   2.641   21.148   41.218
C   3.204   22.306   42.028
C   4.078   21.794   43.180
C   3.597   22.195   44.632
C   4.786   22.472   45.582
C   2.605   21.242   45.251
C   1.168   21.680   44.997
C   0.585   22.346   46.286
C   -0.688   21.571   46.659
C   -1.986   21.953   45.902
C   -0.896   21.401   48.168
H   -2.595   20.991   29.596
H   -1.201   20.468   23.432
H   -0.715   13.996   24.186
H   -2.956   16.688   31.828
H   -2.636   19.508   31.602
H   -4.957   19.482   31.546
H   -4.828   17.761   31.206
H   -4.882   19.028   29.883
H   -0.704   18.577   32.003
H   -0.125   16.926   31.925
H   -2.010   16.619   33.929
H   -1.506   18.305   34.120
H   -2.659   23.698   27.670
H   -0.879   23.905   27.992
H   -1.808   23.103   29.180
H   -0.072   24.609   25.901
H   -1.848   24.682   25.968
H   -0.981   25.076   24.526
H   -0.373   18.284   21.937
H   -1.486   15.585   22.280
H   -3.015   18.882   22.176
H   -2.998   17.260   21.295
H   -2.065   18.618   20.690
H   0.826   16.689   21.218
H   0.607   15.052   21.688
H   1.640   16.638   24.025
H   2.639   17.000   22.646
H   2.416   15.336   23.055
H   -1.332   12.038   24.898
H   -2.083   11.383   26.252
H   -0.338   11.510   26.275
H   -2.984   13.900   30.623
H   0.001   12.388   33.496
H   -1.389   12.558   34.636
H   -0.304   13.861   34.193
H   1.835   17.096   36.540
H   0.745   15.690   36.953
H   0.173   18.472   38.077
H   1.460   15.670   40.688
H   0.936   14.963   39.090
H   -0.130   15.031   40.523
H   -0.891   17.272   41.225
H   -1.080   18.283   39.806
H   1.655   18.643   40.444
H   0.980   18.473   42.071
H   -0.193   20.229   42.077
H   -0.644   20.113   40.427
H   0.653   22.143   41.206
H   2.246   21.452   38.774
H   0.866   22.551   39.071
H   0.548   20.843   38.710
H   3.435   20.870   40.525
H   2.675   20.317   41.922
H   2.426   22.946   42.444
H   3.838   22.890   41.362
H   5.062   22.113   42.837
H   4.117   20.706   43.126
H   3.185   23.203   44.575
H   4.875   21.679   46.325
H   4.650   23.454   46.033
H   5.730   22.407   45.040
H   2.724   21.095   46.324
H   2.895   20.251   44.902
H   0.612   20.798   44.677
H   1.063   22.269   44.086
H   0.248   23.328   45.955
H   1.351   22.408   47.059
H   -0.495   20.544   46.347
H   -1.727   22.273   44.893
H   -2.612   22.732   46.336
H   -2.535   21.013   45.839
H   -0.292   22.142   48.692
H   -0.597   20.462   48.635
H   -1.961   21.511   48.376

