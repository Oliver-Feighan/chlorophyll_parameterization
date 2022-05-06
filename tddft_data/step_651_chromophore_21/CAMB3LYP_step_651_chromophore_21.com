%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_651_chromophore_21 TDDFT with cam-b3lyp functional

0 1
Mg   15.971   52.358   25.297
C   17.744   50.602   28.017
C   13.655   53.070   27.808
C   13.939   53.341   22.885
C   18.062   50.682   23.030
N   15.780   51.797   27.614
C   16.677   51.276   28.478
C   16.250   51.611   29.975
C   14.814   51.991   29.748
C   14.778   52.316   28.264
C   13.804   50.865   30.032
C   17.160   52.681   30.644
C   17.325   52.636   32.135
C   16.496   51.732   33.078
O   16.936   50.880   33.812
O   15.194   52.110   32.965
N   14.126   53.140   25.325
C   13.423   53.461   26.419
C   12.359   54.385   25.989
C   12.449   54.544   24.577
C   13.533   53.635   24.188
C   11.517   55.150   26.975
C   11.650   55.307   23.546
O   11.983   55.511   22.405
C   10.554   56.250   24.144
N   15.997   51.991   23.263
C   15.060   52.570   22.445
C   15.423   52.491   20.950
C   16.823   51.777   21.061
C   17.009   51.477   22.562
C   14.292   51.826   20.032
C   18.013   52.588   20.534
C   18.512   51.874   19.302
N   17.625   51.030   25.483
C   18.351   50.433   24.441
C   19.478   49.715   25.024
C   19.293   49.764   26.404
C   18.035   50.507   26.623
C   20.548   48.991   24.203
C   19.870   49.414   27.670
O   20.940   48.909   27.934
C   18.916   49.917   28.802
C   18.444   48.763   29.586
O   18.081   47.718   29.057
O   18.531   48.928   30.929
C   18.052   47.807   31.728
C   14.409   51.552   34.055
C   13.183   52.361   34.235
C   12.645   53.014   35.292
C   12.958   52.660   36.702
C   11.509   53.998   34.993
C   10.179   53.877   35.910
C   9.921   55.161   36.690
C   9.437   56.282   35.808
C   10.290   57.511   35.985
C   7.953   56.578   36.162
C   7.039   56.644   34.915
C   6.802   57.992   34.269
C   5.370   58.634   34.486
C   4.833   59.290   33.155
C   5.441   59.734   35.606
C   4.261   59.663   36.585
C   4.669   59.099   37.978
C   4.551   57.552   38.250
C   3.196   57.115   38.750
C   5.640   57.126   39.201
H   13.008   53.511   28.569
H   13.367   53.733   22.042
H   18.767   50.307   22.285
H   16.459   50.674   30.490
H   14.495   52.880   30.294
H   12.994   51.234   30.661
H   14.334   50.007   30.445
H   13.327   50.549   29.105
H   16.833   53.695   30.415
H   18.147   52.630   30.183
H   17.123   53.680   32.376
H   18.390   52.461   32.288
H   11.663   56.229   27.017
H   11.659   54.743   27.976
H   10.475   54.997   26.693
H   9.826   55.643   24.683
H   10.147   56.826   23.314
H   11.086   56.978   24.756
H   15.572   53.506   20.581
H   16.777   50.757   20.682
H   13.488   51.613   20.737
H   14.755   50.924   19.631
H   13.965   52.451   19.201
H   18.784   52.786   21.278
H   17.713   53.579   20.195
H   19.579   52.053   19.170
H   18.085   52.496   18.515
H   18.329   50.803   19.216
H   21.445   49.606   24.124
H   20.025   48.777   23.272
H   20.824   48.049   24.677
H   19.508   50.556   29.456
H   17.623   48.240   32.632
H   18.919   47.201   31.992
H   17.241   47.294   31.211
H   14.925   51.641   35.011
H   14.104   50.552   33.747
H   12.865   52.588   33.218
H   12.360   53.250   37.397
H   14.033   52.831   36.739
H   12.764   51.593   36.814
H   11.301   53.826   33.937
H   11.955   54.975   35.176
H   10.225   53.007   36.566
H   9.251   53.690   35.369
H   10.814   55.503   37.213
H   9.291   55.089   37.577
H   9.494   56.089   34.736
H   9.810   58.287   35.388
H   11.289   57.402   35.562
H   10.284   57.719   37.055
H   7.963   57.600   36.541
H   7.457   55.950   36.902
H   6.093   56.106   34.966
H   7.603   56.160   34.118
H   6.955   57.908   33.193
H   7.517   58.649   34.765
H   4.649   57.851   34.725
H   4.881   58.531   32.374
H   5.461   60.164   32.980
H   3.800   59.577   33.350
H   5.539   60.736   35.190
H   6.268   59.389   36.227
H   3.461   58.998   36.261
H   3.798   60.628   36.790
H   3.950   59.557   38.657
H   5.668   59.481   38.190
H   4.726   57.124   37.263
H   2.768   56.574   37.905
H   2.608   57.990   39.024
H   3.339   56.486   39.629
H   5.774   56.061   39.387
H   5.275   57.541   40.141
H   6.653   57.501   39.054

