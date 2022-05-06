%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1301_chromophore_26 TDDFT with blyp functional

0 1
Mg   -8.052   19.364   42.648
C   -4.620   18.788   42.747
C   -7.354   22.717   42.344
C   -11.319   19.898   42.520
C   -8.514   15.876   42.439
N   -6.190   20.554   42.488
C   -4.899   20.086   42.476
C   -3.872   21.164   42.232
C   -4.797   22.482   42.327
C   -6.193   21.920   42.453
C   -4.466   23.485   43.534
C   -3.164   21.151   40.849
C   -3.725   20.351   39.643
C   -3.303   20.846   38.171
O   -2.168   21.142   37.882
O   -4.392   20.905   37.365
N   -9.192   21.089   42.469
C   -8.708   22.335   42.351
C   -9.824   23.237   42.227
C   -10.959   22.439   42.296
C   -10.558   21.063   42.468
C   -9.612   24.690   42.096
C   -12.396   22.939   42.309
O   -12.624   24.101   42.681
C   -13.641   21.984   41.914
N   -9.641   18.116   42.502
C   -10.878   18.607   42.678
C   -11.946   17.414   42.542
C   -10.984   16.167   42.442
C   -9.633   16.743   42.475
C   -12.900   17.471   43.751
C   -11.273   15.299   41.215
C   -11.472   13.849   41.561
N   -6.822   17.591   42.659
C   -7.145   16.267   42.707
C   -5.944   15.467   42.925
C   -4.952   16.468   42.945
C   -5.552   17.678   42.756
C   -5.775   14.050   43.285
C   -3.521   16.674   43.150
O   -2.613   15.832   43.493
C   -3.198   18.206   42.971
C   -2.409   18.665   44.182
O   -1.278   19.091   44.176
O   -3.190   18.518   45.344
C   -2.531   19.020   46.613
C   -4.087   21.438   35.981
C   -5.332   21.391   35.089
C   -5.813   20.303   34.451
C   -5.326   18.874   34.639
C   -6.969   20.497   33.437
C   -8.274   20.434   34.148
C   -9.386   19.926   33.178
C   -10.774   20.635   33.344
C   -11.841   19.584   33.781
C   -11.190   21.483   32.074
C   -10.941   22.985   32.203
C   -9.633   23.406   31.369
C   -10.032   24.360   30.239
C   -9.212   25.623   30.261
C   -9.910   23.742   28.807
C   -11.024   24.325   27.844
C   -10.561   24.382   26.423
C   -11.442   23.576   25.485
C   -11.578   22.022   25.761
C   -10.969   23.747   24.059
H   -7.180   23.777   42.147
H   -12.390   20.031   42.684
H   -8.768   14.823   42.579
H   -3.118   21.175   43.019
H   -4.715   22.964   41.353
H   -5.293   23.879   44.124
H   -4.049   24.374   43.060
H   -3.732   23.115   44.250
H   -2.192   20.665   40.921
H   -3.021   22.172   40.493
H   -4.806   20.240   39.721
H   -3.421   19.326   39.857
H   -10.550   25.030   41.658
H   -8.848   24.982   41.376
H   -9.457   25.248   43.020
H   -14.003   21.440   42.786
H   -13.466   21.240   41.136
H   -14.475   22.623   41.622
H   -12.514   17.670   41.648
H   -11.138   15.525   43.310
H   -13.278   16.456   43.875
H   -13.827   18.029   43.623
H   -12.299   17.690   44.633
H   -10.468   15.532   40.518
H   -12.121   15.669   40.638
H   -12.390   13.505   42.038
H   -10.746   13.500   42.295
H   -11.106   13.213   40.755
H   -5.291   13.645   42.396
H   -6.659   13.448   43.491
H   -4.963   13.878   43.992
H   -2.552   18.280   42.096
H   -3.038   19.953   46.858
H   -1.469   19.256   46.548
H   -2.457   18.247   47.378
H   -3.340   20.811   35.493
H   -3.564   22.394   35.959
H   -5.775   22.350   34.822
H   -4.369   18.827   35.157
H   -5.123   18.446   33.657
H   -6.083   18.294   35.167
H   -6.939   19.711   32.682
H   -6.846   21.428   32.885
H   -8.433   21.412   34.602
H   -8.340   19.694   34.946
H   -9.402   18.841   33.277
H   -8.970   20.151   32.196
H   -10.627   21.338   34.165
H   -12.652   19.649   33.056
H   -12.237   19.910   34.742
H   -11.466   18.560   33.812
H   -12.228   21.315   31.784
H   -10.758   21.033   31.180
H   -10.677   23.243   33.229
H   -11.798   23.605   31.941
H   -9.089   22.600   30.875
H   -8.978   23.903   32.084
H   -11.064   24.703   30.319
H   -9.075   25.859   31.317
H   -9.699   26.424   29.705
H   -8.212   25.525   29.839
H   -9.958   22.654   28.778
H   -8.931   24.039   28.430
H   -11.117   25.364   28.159
H   -12.029   23.918   27.954
H   -9.604   23.861   26.445
H   -10.415   25.426   26.148
H   -12.441   24.012   25.507
H   -11.327   21.759   26.789
H   -10.988   21.471   25.029
H   -12.632   21.754   25.684
H   -9.948   24.124   23.985
H   -11.603   24.422   23.484
H   -10.918   22.833   23.468

