%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_651_chromophore_9 TDDFT with PBE1PBE functional

0 1
Mg   35.461   1.787   29.884
C   33.190   2.752   32.265
C   37.932   1.833   32.421
C   37.861   1.309   27.529
C   32.988   2.026   27.420
N   35.550   2.405   32.055
C   34.437   2.593   32.822
C   34.839   2.707   34.302
C   36.393   2.680   34.333
C   36.679   2.313   32.880
C   37.177   3.844   35.059
C   34.169   1.486   35.175
C   33.692   1.951   36.617
C   34.398   3.142   37.200
O   33.948   4.221   37.223
O   35.556   2.778   37.837
N   37.529   1.480   30.026
C   38.342   1.468   31.132
C   39.706   1.090   30.734
C   39.711   1.047   29.279
C   38.316   1.287   28.897
C   40.806   0.909   31.730
C   40.897   0.809   28.293
O   40.791   0.923   27.083
C   42.190   0.516   28.877
N   35.381   1.550   27.794
C   36.524   1.350   27.044
C   36.155   1.109   25.547
C   34.658   1.331   25.501
C   34.272   1.704   27.020
C   37.050   1.909   24.497
C   33.946   0.091   24.893
C   34.029   -1.117   25.849
N   33.436   2.270   29.846
C   32.558   2.287   28.795
C   31.317   2.738   29.304
C   31.496   2.839   30.711
C   32.818   2.594   30.968
C   29.980   2.989   28.468
C   30.847   3.115   31.928
O   29.643   3.409   32.126
C   31.839   2.945   33.065
C   31.738   4.164   33.952
O   32.063   5.269   33.591
O   31.217   3.835   35.203
C   31.054   4.964   36.171
C   36.266   3.860   38.583
C   37.377   3.414   39.479
C   37.271   2.636   40.589
C   35.960   2.357   41.296
C   38.453   2.337   41.440
C   38.663   0.819   41.509
C   38.894   0.276   42.912
C   40.334   -0.205   43.198
C   40.388   -1.151   44.435
C   41.216   1.134   43.500
C   42.506   1.043   42.739
C   43.619   1.841   43.353
C   44.097   3.132   42.530
C   45.435   2.883   41.752
C   44.260   4.407   43.431
C   44.418   5.789   42.565
C   43.206   6.788   42.462
C   43.285   8.094   43.221
C   41.900   8.699   43.567
C   44.327   9.163   42.644
H   38.732   1.813   33.164
H   38.599   1.061   26.763
H   32.277   2.245   26.621
H   34.502   3.704   34.585
H   36.789   1.876   34.954
H   37.798   3.459   35.867
H   36.509   4.559   35.539
H   37.866   4.342   34.377
H   34.816   0.612   35.241
H   33.296   1.089   34.656
H   33.741   1.110   37.309
H   32.640   2.237   36.596
H   41.274   0.004   31.344
H   40.504   0.738   32.763
H   41.499   1.738   31.594
H   42.372   -0.408   29.426
H   42.398   1.318   29.585
H   42.949   0.613   28.100
H   36.417   0.097   25.239
H   34.446   2.178   24.849
H   37.438   1.141   23.828
H   37.955   2.306   24.957
H   36.579   2.765   24.014
H   34.310   -0.205   23.909
H   32.901   0.378   24.771
H   33.039   -1.153   26.303
H   34.810   -1.034   26.605
H   34.274   -2.033   25.311
H   29.314   2.135   28.593
H   30.175   3.135   27.406
H   29.427   3.790   28.959
H   31.648   2.064   33.679
H   30.099   5.477   36.056
H   31.826   5.716   36.009
H   30.995   4.657   37.215
H   35.535   4.416   39.171
H   36.776   4.629   38.004
H   38.345   3.752   39.109
H   35.115   2.887   40.854
H   36.152   2.711   42.309
H   35.667   1.308   41.269
H   38.388   2.911   42.364
H   39.308   2.713   40.877
H   39.573   0.647   40.933
H   37.886   0.291   40.956
H   38.239   -0.594   42.976
H   38.627   1.017   43.666
H   40.693   -0.834   42.383
H   40.750   -2.118   44.084
H   39.445   -1.548   44.813
H   41.001   -0.935   45.310
H   41.428   1.089   44.569
H   40.638   1.987   43.144
H   42.392   1.532   41.772
H   42.874   0.057   42.457
H   44.474   1.193   43.542
H   43.372   2.261   44.328
H   43.270   3.408   41.876
H   46.343   3.260   42.223
H   45.118   3.474   40.893
H   45.567   1.829   41.507
H   44.997   4.103   44.174
H   43.304   4.600   43.917
H   44.617   5.586   41.513
H   45.278   6.343   42.940
H   42.345   6.241   42.844
H   42.983   7.100   41.441
H   43.582   7.927   44.257
H   41.291   7.842   43.856
H   41.403   9.302   42.807
H   41.981   9.353   44.435
H   43.882   10.062   42.217
H   45.026   8.771   41.905
H   44.907   9.573   43.471

