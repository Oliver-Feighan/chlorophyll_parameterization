%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1451_chromophore_9 TDDFT with cam-b3lyp functional

0 1
Mg   35.462   1.316   30.278
C   33.294   2.592   32.760
C   38.058   1.486   32.629
C   37.539   0.531   27.875
C   32.818   1.899   27.988
N   35.616   1.992   32.491
C   34.565   2.282   33.274
C   35.058   2.430   34.769
C   36.558   2.257   34.624
C   36.792   1.918   33.147
C   37.377   3.494   35.109
C   34.381   1.185   35.620
C   33.933   1.584   37.033
C   34.756   2.653   37.750
O   34.363   3.820   37.935
O   35.982   2.167   38.126
N   37.522   1.061   30.265
C   38.412   1.152   31.287
C   39.717   0.676   30.841
C   39.661   0.518   29.458
C   38.211   0.676   29.145
C   40.806   0.342   31.762
C   40.764   0.206   28.441
O   40.580   0.222   27.215
C   42.098   -0.126   28.925
N   35.183   1.207   28.217
C   36.249   0.835   27.436
C   35.904   0.806   25.989
C   34.323   1.034   26.000
C   34.043   1.411   27.472
C   36.780   1.604   24.981
C   33.449   -0.113   25.379
C   32.946   -1.158   26.411
N   33.476   1.981   30.282
C   32.592   2.270   29.307
C   31.302   2.804   29.866
C   31.575   2.878   31.238
C   32.867   2.375   31.416
C   30.020   3.012   29.066
C   31.007   3.329   32.499
O   29.945   3.800   32.775
C   32.086   3.059   33.567
C   32.369   4.395   34.168
O   33.139   5.194   33.618
O   31.781   4.570   35.340
C   31.867   5.841   35.952
C   36.737   3.096   38.922
C   37.914   2.376   39.432
C   38.018   1.787   40.626
C   36.886   1.636   41.653
C   39.356   1.091   41.015
C   40.495   1.930   41.606
C   40.730   1.629   43.092
C   41.664   0.449   43.419
C   40.969   -0.807   44.058
C   42.885   0.930   44.194
C   44.201   0.486   43.450
C   45.382   1.339   43.838
C   45.766   2.397   42.734
C   47.007   2.053   41.970
C   45.851   3.847   43.341
C   44.462   4.512   43.250
C   44.318   5.437   41.979
C   44.028   7.005   42.301
C   45.104   7.681   43.292
C   42.581   7.182   42.829
H   38.767   1.548   33.457
H   38.276   0.130   27.175
H   32.002   2.168   27.314
H   34.856   3.461   35.060
H   36.958   1.416   35.191
H   38.132   3.294   35.869
H   36.807   4.311   35.552
H   37.921   3.944   34.278
H   35.121   0.431   35.888
H   33.600   0.631   35.099
H   33.965   0.715   37.691
H   32.896   1.907   36.945
H   40.442   0.601   32.756
H   41.768   0.837   31.630
H   40.973   -0.732   31.835
H   42.569   0.842   29.101
H   42.445   -0.642   28.030
H   42.115   -0.855   29.735
H   35.991   -0.261   25.784
H   34.100   1.903   25.380
H   37.255   0.915   24.282
H   37.546   2.088   25.587
H   36.252   2.373   24.417
H   34.119   -0.605   24.675
H   32.607   0.221   24.774
H   31.859   -1.236   26.430
H   33.301   -0.926   27.415
H   33.283   -2.161   26.149
H   29.181   3.057   29.759
H   29.788   2.293   28.280
H   30.124   3.979   28.573
H   31.754   2.463   34.418
H   30.896   6.112   36.367
H   32.108   6.703   35.330
H   32.648   5.854   36.712
H   36.260   3.476   39.825
H   37.102   3.994   38.423
H   38.839   2.363   38.857
H   37.430   1.684   42.597
H   36.373   0.682   41.537
H   36.162   2.450   41.606
H   39.923   0.612   40.217
H   39.076   0.289   41.697
H   40.142   2.960   41.562
H   41.461   1.757   41.132
H   39.753   1.521   43.564
H   41.115   2.583   43.452
H   41.908   0.022   42.446
H   41.691   -1.604   44.239
H   40.279   -1.251   43.341
H   40.438   -0.585   44.984
H   42.916   0.449   45.172
H   42.979   2.009   44.320
H   44.025   0.292   42.392
H   44.427   -0.490   43.880
H   46.165   0.595   43.985
H   45.040   1.849   44.739
H   45.000   2.353   41.961
H   47.844   2.740   42.097
H   46.903   1.937   40.892
H   47.246   1.070   42.375
H   46.543   4.361   42.673
H   46.068   3.896   44.408
H   44.269   4.998   44.206
H   43.779   3.665   43.186
H   43.481   5.134   41.350
H   45.222   5.394   41.372
H   43.960   7.473   41.318
H   45.669   7.001   43.929
H   44.740   8.495   43.918
H   45.907   8.117   42.698
H   42.078   6.218   42.906
H   42.138   7.755   42.014
H   42.469   7.796   43.723

