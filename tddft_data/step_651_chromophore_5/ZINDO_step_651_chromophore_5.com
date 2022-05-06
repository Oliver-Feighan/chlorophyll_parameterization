%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_651_chromophore_5 ZINDO

0 1
Mg   24.357   -7.456   46.291
C   26.381   -4.781   45.189
C   21.748   -6.014   44.451
C   22.628   -10.246   46.476
C   27.363   -8.956   47.241
N   24.089   -5.618   45.082
C   25.026   -4.658   44.762
C   24.446   -3.522   43.922
C   22.954   -3.877   43.962
C   22.915   -5.242   44.554
C   21.988   -2.845   44.652
C   25.035   -3.474   42.480
C   24.913   -2.059   41.723
C   24.081   -2.031   40.422
O   22.983   -1.484   40.216
O   24.816   -2.659   39.407
N   22.374   -7.947   45.777
C   21.476   -7.243   45.043
C   20.249   -8.017   44.843
C   20.497   -9.235   45.519
C   21.912   -9.189   46.024
C   19.102   -7.613   44.065
C   19.389   -10.308   45.778
O   18.289   -10.242   45.227
C   19.682   -11.450   46.716
N   24.905   -9.379   46.769
C   24.008   -10.386   46.875
C   24.587   -11.669   47.601
C   26.090   -11.114   47.887
C   26.169   -9.734   47.164
C   23.814   -11.993   48.911
C   27.308   -12.046   47.462
C   28.496   -12.319   48.454
N   26.430   -7.011   46.311
C   27.464   -7.661   46.810
C   28.650   -6.899   46.916
C   28.291   -5.716   46.208
C   26.940   -5.817   45.916
C   29.914   -7.363   47.534
C   28.697   -4.445   45.766
O   29.801   -3.907   45.949
C   27.511   -3.729   45.080
C   27.231   -2.540   45.895
O   27.605   -1.388   45.622
O   26.420   -2.783   46.991
C   25.867   -1.535   47.543
C   24.374   -2.696   37.979
C   24.236   -4.145   37.577
C   24.503   -4.753   36.401
C   25.098   -3.922   35.169
C   24.321   -6.244   36.258
C   23.006   -6.668   35.507
C   23.199   -7.278   34.121
C   22.500   -6.445   32.975
C   21.031   -6.851   32.837
C   23.333   -6.664   31.687
C   23.898   -5.352   31.205
C   23.017   -4.451   30.306
C   23.428   -4.461   28.763
C   22.327   -3.878   27.830
C   24.756   -3.701   28.603
C   25.752   -4.439   27.664
C   25.685   -3.990   26.242
C   24.976   -5.203   25.465
C   26.082   -6.195   24.784
C   24.019   -4.632   24.317
H   21.037   -5.389   43.908
H   22.117   -11.201   46.613
H   28.224   -9.363   47.775
H   24.601   -2.537   44.364
H   22.554   -3.943   42.950
H   21.197   -2.658   43.926
H   22.517   -1.901   44.785
H   21.519   -3.263   45.543
H   24.364   -4.109   41.902
H   26.059   -3.848   42.441
H   25.917   -1.768   41.414
H   24.591   -1.315   42.451
H   18.434   -8.435   43.808
H   19.483   -7.223   43.121
H   18.535   -6.820   44.552
H   20.292   -11.024   47.512
H   20.270   -12.213   46.206
H   18.775   -11.946   47.063
H   24.526   -12.547   46.957
H   26.200   -10.843   48.937
H   24.545   -12.049   49.717
H   23.261   -12.910   48.706
H   23.083   -11.196   49.051
H   27.777   -11.539   46.618
H   26.835   -13.012   47.286
H   28.193   -12.104   49.479
H   29.198   -11.556   48.121
H   28.820   -13.360   48.427
H   30.295   -8.072   46.799
H   29.739   -7.856   48.490
H   30.564   -6.497   47.665
H   27.753   -3.342   44.090
H   25.538   -0.950   46.685
H   26.523   -0.958   48.194
H   25.104   -1.888   48.236
H   25.193   -2.291   37.385
H   23.377   -2.292   37.805
H   23.811   -4.787   38.349
H   25.590   -4.620   34.491
H   25.680   -3.079   35.541
H   24.182   -3.625   34.659
H   24.349   -6.716   37.240
H   25.108   -6.722   35.674
H   22.381   -5.775   35.508
H   22.395   -7.358   36.089
H   22.853   -8.309   34.184
H   24.258   -7.372   33.883
H   22.479   -5.424   33.354
H   20.868   -7.319   31.866
H   20.325   -6.034   32.984
H   20.749   -7.581   33.595
H   22.737   -7.102   30.886
H   24.161   -7.353   31.852
H   24.775   -5.610   30.611
H   24.409   -4.880   32.044
H   22.993   -3.475   30.791
H   22.036   -4.919   30.232
H   23.605   -5.508   28.517
H   22.720   -2.986   27.343
H   21.485   -3.617   28.471
H   21.890   -4.578   27.117
H   25.222   -3.724   29.589
H   24.594   -2.668   28.298
H   25.677   -5.515   27.822
H   26.689   -4.070   28.081
H   26.691   -3.841   25.848
H   25.130   -3.052   26.232
H   24.419   -5.838   26.154
H   27.030   -5.700   24.570
H   25.662   -6.747   23.943
H   26.264   -6.898   25.597
H   23.019   -5.028   24.493
H   24.159   -5.031   23.312
H   23.844   -3.560   24.408

