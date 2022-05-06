%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1551_chromophore_13 TDDFT with blyp functional

0 1
Mg   46.487   25.012   29.361
C   47.032   27.474   31.788
C   45.650   22.894   31.797
C   46.315   22.658   27.025
C   47.391   27.368   26.903
N   46.525   25.190   31.619
C   46.512   26.296   32.365
C   45.974   26.163   33.804
C   45.789   24.577   33.839
C   45.922   24.230   32.304
C   46.745   23.792   34.801
C   44.780   27.013   34.179
C   44.949   28.050   35.264
C   44.009   28.033   36.455
O   42.903   28.614   36.504
O   44.570   27.162   37.400
N   45.949   23.131   29.419
C   45.699   22.381   30.499
C   45.371   21.044   30.145
C   45.598   21.008   28.767
C   45.959   22.294   28.328
C   44.916   20.064   31.168
C   45.531   19.724   27.829
O   45.732   19.604   26.596
C   45.002   18.421   28.372
N   46.958   24.918   27.257
C   46.844   23.812   26.524
C   46.958   24.077   24.997
C   47.050   25.669   24.987
C   47.100   26.052   26.488
C   48.152   23.340   24.445
C   45.839   26.289   24.274
C   44.590   26.824   25.126
N   47.223   26.949   29.320
C   47.484   27.768   28.268
C   48.024   29.054   28.727
C   47.841   28.995   30.085
C   47.341   27.675   30.389
C   48.422   30.197   27.903
C   47.986   29.642   31.331
O   48.543   30.704   31.619
C   47.355   28.723   32.498
C   48.390   28.466   33.517
O   49.437   27.819   33.289
O   48.254   29.224   34.657
C   49.194   28.905   35.683
C   43.730   27.069   38.631
C   44.172   25.826   39.389
C   43.817   25.620   40.672
C   42.932   26.562   41.590
C   44.285   24.369   41.498
C   43.531   23.022   41.032
C   42.722   22.316   42.151
C   41.307   21.888   41.750
C   41.228   21.064   40.503
C   40.213   23.021   41.862
C   39.514   23.076   43.330
C   39.993   24.258   44.235
C   40.005   23.818   45.760
C   38.797   24.455   46.521
C   41.314   24.132   46.439
C   42.043   22.973   47.151
C   43.473   22.611   46.726
C   44.656   23.587   47.140
C   45.600   24.078   46.010
C   45.443   22.867   48.292
H   45.303   22.176   32.543
H   46.164   21.884   26.269
H   47.661   28.096   26.136
H   46.783   26.520   34.442
H   44.873   24.318   34.369
H   46.293   22.953   35.328
H   47.178   24.553   35.450
H   47.537   23.297   34.239
H   43.938   26.355   34.391
H   44.551   27.534   33.250
H   44.624   29.024   34.901
H   45.969   28.061   35.648
H   45.617   20.184   31.994
H   45.058   19.013   30.914
H   43.951   20.288   31.621
H   44.079   18.459   28.951
H   45.826   18.023   28.964
H   44.814   17.674   27.602
H   46.020   23.719   24.573
H   47.920   26.089   24.481
H   48.932   24.005   24.073
H   47.707   22.846   23.581
H   48.462   22.552   25.131
H   45.428   25.653   23.490
H   46.383   27.150   23.886
H   44.722   26.835   26.208
H   43.708   26.284   24.781
H   44.379   27.887   25.001
H   48.480   30.005   26.832
H   49.428   30.488   28.202
H   47.758   31.039   28.097
H   46.456   29.028   33.033
H   50.243   28.862   35.388
H   49.009   27.915   36.098
H   49.171   29.619   36.506
H   42.641   27.075   38.590
H   44.015   27.882   39.298
H   44.797   25.028   38.989
H   42.082   25.969   41.927
H   42.548   27.428   41.051
H   43.668   26.926   42.306
H   44.225   24.626   42.556
H   45.341   24.203   41.284
H   44.324   22.348   40.709
H   42.898   23.333   40.201
H   42.609   23.027   42.970
H   43.273   21.522   42.654
H   41.182   21.091   42.483
H   42.225   20.979   40.070
H   40.597   21.525   39.743
H   40.933   20.046   40.760
H   39.410   22.808   41.157
H   40.559   24.036   41.665
H   39.637   22.098   43.797
H   38.480   23.246   43.033
H   39.212   25.009   44.114
H   41.019   24.475   43.938
H   39.908   22.738   45.861
H   38.206   23.590   46.823
H   38.169   25.127   45.937
H   39.046   25.062   47.391
H   41.171   24.802   47.286
H   42.028   24.692   45.834
H   41.517   22.028   47.016
H   42.019   23.284   48.196
H   43.453   22.622   45.636
H   43.713   21.612   47.090
H   44.150   24.387   47.680
H   45.307   23.677   45.040
H   46.666   23.903   46.161
H   45.516   25.165   46.016
H   46.513   22.847   48.085
H   45.208   21.844   48.587
H   45.432   23.457   49.208

